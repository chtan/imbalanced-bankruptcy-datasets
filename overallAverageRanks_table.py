import os
import sys
import time
import random
import logging
import warnings
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    SelectPercentile, chi2, f_classif, VarianceThreshold, mutual_info_classif
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score

from deap import base, creator, tools, algorithms
import pyswarms as ps
import joblib

from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler



#
# Genetic Algorithm (GA)
#
class GeneticFeatureSelector(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that performs feature selection
    using a Genetic Algorithm (via the DEAP library).
    """
    def __init__(self, n_features_to_select=5, population_size=100, generations=50,
                 patience=10, cv=5, random_state=42, verbose=True):

        # Number of features the GA must select
        self.n_features_to_select = n_features_to_select

        # Number of candidate solutions (feature subsets) per generation
        self.population_size = population_size

        # Maximum number of generations the GA will run
        self.generations = generations

        # Early stopping: stop if no improvement after this many generations
        self.patience = patience

        # Cross-validation folds used to evaluate a feature subset
        self.cv = cv

        # Random seed for reproducibility
        self.random_state = random_state

        # Whether to print progress information
        self.verbose = verbose


    # ---------------------------------------------------------
    # INITIALIZE DEAP COMPONENTS (GENETIC ALGORITHM SETUP)
    # ---------------------------------------------------------
    def _init_deap(self):

        # Fix random seed for reproducibility
        random.seed(self.random_state)

        # Total number of available features in dataset
        self.n_total_features = self.X_.shape[1]

        # DEAP uses a global "creator" registry.
        # We ensure these objects are created only once.
        if not hasattr(self, '_creator_initialized'):

            # Define a fitness type that should be maximized
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))

            # Define an Individual = list with attached FitnessMax
            creator.create("Individual", list, fitness=creator.FitnessMax)

            self._creator_initialized = True

        # Toolbox is DEAP’s container for genetic operators
        self.toolbox = base.Toolbox()


        # ---------------------------------------------------------
        # INDIVIDUAL CREATION
        # ---------------------------------------------------------
        def init_individual():

            # Individual = binary list indicating feature selection
            # Example: [0,1,0,1,0] means select features 1 and 3
            ind = [0] * self.n_total_features

            # Randomly choose exactly n_features_to_select positions
            selected = random.sample(
                range(self.n_total_features), 
                self.n_features_to_select
            )

            # Set those positions to 1
            for i in selected:
                ind[i] = 1

            # Convert to DEAP Individual object
            return creator.Individual(ind)

        # Register function that creates a single individual
        self.toolbox.register("individual", init_individual)

        # Register function that creates a population (list of individuals)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)


        # ---------------------------------------------------------
        # FITNESS CACHE
        # ---------------------------------------------------------
        # Avoid re-evaluating identical feature subsets
        self.fitness_cache = {}


        # ---------------------------------------------------------
        # FITNESS FUNCTION
        # ---------------------------------------------------------
        def evaluate(ind):

            # Convert individual to tuple so it can be used as dictionary key
            key = tuple(ind)

            # If already evaluated before, reuse cached score
            if key in self.fitness_cache:
                return self.fitness_cache[key]

            # Determine which features are selected (bit == 1)
            selected = [i for i, bit in enumerate(ind) if bit == 1]

            # If no features selected -> invalid solution
            if len(selected) == 0:
                return 0.,

            # Train logistic regression on only selected features
            # and evaluate using cross-validation
            score = cross_val_score(
                LogisticRegression(
                    solver='liblinear',
                    max_iter=2000,
                    random_state=self.random_state
                ),
                self.X_[:, selected],   # subset of features
                self.y_,
                cv=self.cv
            ).mean()

            # Store in cache
            self.fitness_cache[key] = (score,)

            return score,

        # Register evaluation function
        self.toolbox.register("evaluate", evaluate)


        # ---------------------------------------------------------
        # CROSSOVER (MATING)
        # ---------------------------------------------------------
        def mate(ind1, ind2):

            # Perform two-point crossover between parents
            tools.cxTwoPoint(ind1, ind2)

            # Ensure children still select exactly n_features_to_select
            for ind in [ind1, ind2]:

                while sum(ind) != self.n_features_to_select:

                    # Indices of selected features
                    ones = [i for i, x in enumerate(ind) if x == 1]

                    # Indices of unselected features
                    zeros = [i for i, x in enumerate(ind) if x == 0]

                    # Too many features selected -> turn one off
                    if sum(ind) > self.n_features_to_select:
                        ind[random.choice(ones)] = 0

                    # Too few features selected -> turn one on
                    elif sum(ind) < self.n_features_to_select:
                        ind[random.choice(zeros)] = 1

            return ind1, ind2


        # ---------------------------------------------------------
        # MUTATION
        # ---------------------------------------------------------
        def mutate(ind):

            # Find positions with selected features
            ones = [i for i, bit in enumerate(ind) if bit == 1]

            # Find positions with unselected features
            zeros = [i for i, bit in enumerate(ind) if bit == 0]

            # Swap one selected feature with one unselected feature
            if ones and zeros:
                i1, i0 = random.choice(ones), random.choice(zeros)
                ind[i1], ind[i0] = 0, 1

            return ind,


        # Register genetic operators
        self.toolbox.register("mate", mate)
        self.toolbox.register("mutate", mutate)

        # Selection method: tournament selection
        self.toolbox.register("select", tools.selTournament, tournsize=3)



    # ---------------------------------------------------------
    # TRAINING PHASE (RUN GENETIC ALGORITHM)
    # ---------------------------------------------------------
    def fit(self, X, y):

        # Store training data as numpy arrays
        self.X_ = np.array(X)
        self.y_ = np.array(y)

        # Initialize DEAP toolbox and operators
        self._init_deap()

        # Create initial random population
        pop = self.toolbox.population(n=self.population_size)

        # Track best score found
        best_fitness = -np.inf

        # Counter for early stopping
        stagnation = 0


        # Main GA loop (generations)
        for gen in range(1, self.generations + 1):

            # Apply crossover and mutation to produce offspring
            offspring = algorithms.varAnd(pop, self.toolbox, cxpb=0.5, mutpb=0.2)

            # Identify individuals that have not been evaluated yet
            invalid = [ind for ind in offspring if not ind.fitness.valid]

            # Compute fitness for them
            fitnesses = list(map(self.toolbox.evaluate, invalid))

            # Attach fitness scores
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            # Select next generation population
            pop = self.toolbox.select(offspring, k=len(pop))

            # Find best individual in population
            best_ind = tools.selBest(pop, 1)[0]
            best_score = best_ind.fitness.values[0]

            # Print progress
            if self.verbose:
                print(f"Generation {gen:>2}: Best CV score = {best_score:.4f}")

            # Check improvement
            if best_score > best_fitness:

                best_fitness = best_score
                self.best_individual_ = best_ind

                stagnation = 0

            else:

                stagnation += 1

                # Early stopping if no improvement
                if stagnation >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at generation {gen} (no improvement for {self.patience} generations).")
                    break


        # Extract indices of selected features from best individual
        self.selected_features_ = [i for i, bit in enumerate(self.best_individual_) if bit == 1]

        # Store best score
        self.best_score_ = best_fitness

        # Boolean mask for sklearn compatibility
        self.support_ = np.zeros(self.n_total_features, dtype=bool)
        self.support_[self.selected_features_] = True

        return self


    # ---------------------------------------------------------
    # APPLY FEATURE SELECTION TO NEW DATA
    # ---------------------------------------------------------
    def transform(self, X):
        # Keep only the selected columns
        return X[:, self.support_]


    # ---------------------------------------------------------
    # RETURN FEATURE MASK
    # ---------------------------------------------------------
    def get_support(self, indices=False):
        """
        Return selected features.

        indices=False -> boolean mask of selected features
        indices=True  -> integer indices of selected features
        """
        return np.where(self.support_)[0] if indices else self.support_

#
# Particle Swarm Optimization (PSO)
#
class PSOFeatureSelector(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that performs feature selection
    using Particle Swarm Optimization (PSO).
    """
    def __init__(self, n_select=5, n_particles=20, iters=50,
                 pso_options=None, estimator=None, cv=5, random_state=42):

        # Number of features to select
        self.n_select = n_select

        # Number of particles in the swarm (candidate solutions)
        self.n_particles = n_particles

        # Number of optimization iterations
        self.iters = iters

        # Number of folds used for cross-validation when evaluating a feature subset
        self.cv = cv

        # Random seed for reproducibility
        self.random_state = random_state

        # PSO hyperparameters:
        # c1 = cognitive coefficient (particle's own best influence)
        # c2 = social coefficient (global best influence)
        # w  = inertia weight (momentum of movement)
        self.pso_options = pso_options or {'c1': 1.5, 'c2': 1.5, 'w': 0.7}

        # Base estimator used to evaluate feature subsets
        # Default: Logistic Regression
        self.estimator = estimator or LogisticRegression(
            solver='liblinear',
            max_iter=2000,
            random_state=random_state
        )

        # Stores indices of the selected features after fitting
        self.selected_indices_ = None


    # ---------------------------------------------------------
    # OBJECTIVE FUNCTION FOR PSO
    # ---------------------------------------------------------
    # PSO works by minimizing a cost function. This function evaluates
    # the quality of each particle (candidate feature ranking).
    def _objective_function(self, particles):

        scores = []

        # Each particle represents a vector of feature "weights"
        # Shape: (n_particles, n_features)
        for particle in particles:

            # Rank features according to particle values
            # Select the indices of the top-n features
            top_indices = np.argsort(particle)[-self.n_select:]

            # Convert selected indices to a boolean mask
            mask = np.zeros_like(particle, dtype=bool)
            mask[top_indices] = True

            # Evaluate the selected feature subset using cross-validation
            score = cross_val_score(
                self.estimator,
                self.X_[:, mask],  # use only selected features
                self.y_,
                cv=self.cv
            ).mean()

            # PSO minimizes the objective function,
            # so we negate the score (since higher accuracy is better)
            scores.append(-score)

        # Return cost for each particle
        return np.array(scores)


    # ---------------------------------------------------------
    # FIT METHOD (RUN PSO OPTIMIZATION)
    # ---------------------------------------------------------
    def fit(self, X, y):

        # Store training data
        self.X_ = X
        self.y_ = y

        # Total number of features
        n_features = X.shape[1]

        # Initialize PSO optimizer from pyswarms
        optimizer = ps.single.GlobalBestPSO(

            # Number of particles exploring the search space
            n_particles=self.n_particles,

            # Dimensionality of the problem (one dimension per feature)
            dimensions=n_features,

            # PSO hyperparameters
            options=self.pso_options
        )

        # Run the PSO optimization
        # cost = best objective value found
        # pos  = best particle position (feature weight vector)
        cost, pos = optimizer.optimize(
            self._objective_function,
            iters=self.iters,
            verbose=False
        )

        # After optimization, select the top-n ranked features
        # according to the best particle position
        self.selected_indices_ = np.argsort(pos)[-self.n_select:]

        return self


    # ---------------------------------------------------------
    # APPLY FEATURE SELECTION TO NEW DATA
    # ---------------------------------------------------------
    def transform(self, X):
        # Keep only the selected columns
        return X[:, self.selected_indices_]


    # ---------------------------------------------------------
    # RETURN FEATURE MASK
    # ---------------------------------------------------------
    def get_support(self, indices=False):
        """
        Return selected features.

        indices=False -> boolean mask of selected features
        indices=True  -> integer indices of selected features
        """

        # Create boolean mask of selected features
        mask = np.zeros(self.X_.shape[1], dtype=bool)
        mask[self.selected_indices_] = True

        # Return indices or mask
        return self.selected_indices_ if indices else mask


#
# Feature Selectors and Resamplers
#

# Filter methods
score_funcs = [
    ("Anova", f_classif),
    ("Info", mutual_info_classif),
]

# Wrapper, Embedded
fsmethods_embedded = [
    ("LogisticRegressionWitL1Penalty", LogisticRegression(
        penalty='l1', solver='liblinear', max_iter=2000, random_state=42)),
    ("RandomForest", RandomForestClassifier(random_state=42)),
]

# GA, PSO
fsmethods_wrapper = [
    ("PSO", PSOFeatureSelector(n_select=5, iters=30)),
    ("GA", GeneticFeatureSelector(n_features_to_select=5, generations=30, verbose=False)),
]

# --- Balancing Methods ---
blmethods = [
    ("SMOTE", SMOTE(random_state=42)),
    ("RandomOverSampler", RandomOverSampler(random_state=42)),
    ("RandomUnderSampler", RandomUnderSampler(random_state=42)),
]

# --- Classifiers ---
clmethods = [
    ("RandomForestClassifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
    ("LinearSVC", SVC(kernel='linear', n_jobs=-1)),
    ("SVC", SVC(kernel='rbf', C=1.0, gamma='scale', n_jobs=-1)),
    ("GradientBoostingClassifier", GradientBoostingClassifier(random_state=42)),
    ("KNNClassifier", KNeighborsClassifier(n_jobs=-1)),
    ("NaiveBayes", GaussianNB()),
    ("CART", DecisionTreeClassifier(random_state=42)),
]


# Varying the percentiles for feature selection
percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

output_d = {
    "polish": {},
    "russian": {}, 
    "taiwanese": {}
}

dotplots = {
    "polish": {},
    "russian": {}, 
    "taiwanese": {}
}

# Notes on the datasets:
# 1) sum(datasets['X_russian']['I3']) is 9, almost all entries are 0; I3 is the field "unreliable supplier".
# 2) The Taiwanese data has a column of constant - this triggers a warning, which is harmless.
datasets = joblib.load('datasets.jbl')

for measure in ['roc_aucs', 'f1_scores', 'g_means']:
    for country in ["polish", "russian", "taiwanese"]:
        #
        # Read datasets
        #
        X = datasets[f'X_{country}']
        y = datasets[f'y_{country}'].squeeze() # converts 1d dataframe to series

        if country == "polish":
            mask = X["year"] == 5.0
            X = X[mask].drop(columns=["year"])
            y = y[mask]


        #
        # Cross-validation setup
        #
        cv1 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        d = {
            '10 fold splits': [(train_idx, test_idx) for train_idx, test_idx in cv1.split(X, y)],
        }
        joblib.dump(d, f'cvs_{country}.jbl')


        #Read data into m
        m = {}


        #
        # Pipeline 1
        # fs+bal, switch False
        #
        fsmethods = fsmethods_embedded
        for cvi, (train_idx, test_idx) in enumerate(joblib.load(f'cvs_{country}.jbl')['10 fold splits']):
            cv_name = f"Fold{cvi}"
            switch = False # (fs+bal)
            for fsmethod_name, fsmethod in fsmethods[:]:
                for percentile in percentiles[:]:
                    pct_name = f"Pct{percentile}"
                    for blmethod_name, blmethod in blmethods[:]:
                        for clmethod_name, clmethod in clmethods[:]:
                            name = "_".join([
                                cv_name,  
                                pct_name, 
                                fsmethod_name, 
                                blmethod_name, 
                                clmethod_name, 
                                str(switch) 
                            ])
                            # e.g. name:
                            # Fold0_Pct10_LogisticRegressionWitL1Penalty_RandomOverSampler_CART_False
                            metrics = joblib.load(f"outsv2/outv2_1/{country}/metrics-{name}.jbl")
                            m["1_" + name] = metrics

        #
        # 2
        # bal+fs, switch True
        #
        fsmethods = fsmethods_embedded
        for cvi, (train_idx, test_idx) in enumerate(joblib.load(f'cvs_{country}.jbl')['10 fold splits'][:]):
            cv_name = f"Fold{cvi}"
            switch = True # (bal+fs)
            for blmethod_name, blmethod in blmethods[:]:
                for fsmethod_name, fsmethod in fsmethods[:]:
                    for percentile in percentiles[:]:
                        pct_name = f"Pct{percentile}"
                        for clmethod_name, clmethod in clmethods[:]:
                            name = "_".join([
                                cv_name,
                                pct_name,
                                fsmethod_name,
                                blmethod_name,
                                clmethod_name,
                                str(switch)
                            ])
                            # e.g. name:
                            # Fold0_Pct10_LogisticRegressionWitL1Penalty_RandomOverSampler_CART_True
                            metrics = joblib.load(f"outsv2/outv2_2/{country}/metrics-{name}.jbl")
                            m["2_" + name] = metrics

        #
        # 3
        # filter (both switches)
        #
        for cvi, (train_idx, test_idx) in enumerate(joblib.load(f'cvs_{country}.jbl')['10 fold splits']):
            cv_name = f"Fold{cvi}"
            for percentile in percentiles[:]:
                for score_func_name, score_func in score_funcs[:]:
                    for blmethod_name, blmethod in blmethods[:]:
                        for switch in [True, False][:]:
                            for clmethod_name, clmethod in clmethods[:]:
                                pct_name = f"Pct{percentile}"
                                name = "_".join([
                                    cv_name,
                                    pct_name,
                                    f"filterMethod-{score_func_name}",
                                    blmethod_name,
                                    clmethod_name,
                                    str(switch)
                                ])
                                # e.g. name
                                # metrics-Fold0_Pct10_filterMethod-Anova_RandomOverSampler_CART_False
                                metrics = joblib.load(f"outsv2/outv2_3/{country}/metrics-{name}.jbl")
                                m["3_" + name] = metrics

        #
        # 4
        # fs + bal, switch False
        #
        fsmethods = fsmethods_wrapper
        for cvi, (train_idx, test_idx) in enumerate(joblib.load(f'cvs_{country}.jbl')['10 fold splits']):
            cv_name = f"Fold{cvi}"
            switch = False # (fs+bal)
            for fsmethod_name, fsmethod in fsmethods[:]:
                for percentile in percentiles[:]:
                    pct_name = f"Pct{percentile}"
                    for blmethod_name, blmethod in blmethods[:]:
                        for clmethod_name, clmethod in clmethods[:]:
                            name = "_".join([
                                cv_name,
                                pct_name,
                                fsmethod_name,
                                blmethod_name,
                                clmethod_name,
                                str(switch)
                            ])
                            # e.g. name
                            # Fold0_Pct10_GA_RandomOverSampler_CART_False
                            metrics = joblib.load(f"outsv2/outv2_4/{country}/metrics-{name}.jbl")
                            m["4_" + name] = metrics

        #
        # 5
        # bal + fs, switch True
        #
        fsmethods = fsmethods_wrapper
        for cvi, (train_idx, test_idx) in enumerate(joblib.load(f'cvs_{country}.jbl')['10 fold splits']):
            cv_name = f"Fold{cvi}"
            switch = True # (bal+fs)
            for blmethod_name, blmethod in blmethods[:]:
                for fsmethod_name, fsmethod in fsmethods[:]:   
                    for percentile in percentiles[:]:
                        pct_name = f"Pct{percentile}"
                        for clmethod_name, clmethod in clmethods[:]:
                            name = "_".join([
                                cv_name,
                                pct_name,
                                fsmethod_name,
                                blmethod_name,
                                clmethod_name,
                                str(switch)
                            ])
                            # e.g. name:
                            # Fold0_Pct10_GA_RandomOverSampler_CART_True
                            metrics = joblib.load(f"outsv2/outv2_5/{country}/metrics-{name}.jbl")
                            m["5_" + name] = metrics
  
        #
        # 6
        # fs, embedded
        #
        fsmethods = fsmethods_embedded
        for cvi, (train_idx, test_idx) in enumerate(joblib.load(f'cvs_{country}.jbl')['10 fold splits']):
            cv_name = f"Fold{cvi}"
            switch = True # (bal+fs)
            for blmethod_name, blmethod in blmethods[:1]: # no balance, so just pick one             
                for fsmethod_name, fsmethod in fsmethods[:]:
                    for percentile in percentiles[:]:
                        pct_name = f"Pct{percentile}"
                        for clmethod_name, clmethod in clmethods[:]:
                            name = "_".join([
                                cv_name,
                                pct_name,
                                fsmethod_name,
                                "---",
                                clmethod_name,
                                str(switch)
                            ])
                            # e.g. name
                            # metrics-Fold0_Pct10_LogisticRegressionWitL1Penalty_---_CART_True
                            metrics = joblib.load(f"outsv2/outv2_6/{country}/metrics-{name}.jbl")
                            m["6_" + name] = metrics


        #
        # 7
        # fs, filter
        #
        for cvi, (train_idx, test_idx) in enumerate(joblib.load(f'cvs_{country}.jbl')['10 fold splits']):
            cv_name = f"Fold{cvi}"
            for percentile in percentiles[:]:
                for score_func_name, score_func in score_funcs[:]:
                    for blmethod_name, blmethod in blmethods[:1]: # this is irrelevant
                        for switch in [True, False][:1]:          # this too; just set to true
                            for clmethod_name, clmethod in clmethods[:]:
                                pct_name = f"Pct{percentile}"
                                name = "_".join([
                                    cv_name,
                                    pct_name,
                                    score_func_name,
                                    "---",
                                    clmethod_name,
                                    str(switch)
                                ])
                                # e.g. name
                                # metrics-Fold0_Pct10_Anova_---_CART_True
                                metrics = joblib.load(f"outsv2/outv2_7/{country}/metrics-{name}.jbl")
                                m["7_" + name] = metrics

        #
        # 8
        # fs, wrapper
        #
        fsmethods = fsmethods_wrapper
        for cvi, (train_idx, test_idx) in enumerate(joblib.load(f'cvs_{country}.jbl')['10 fold splits']):
            cv_name = f"Fold{cvi}"
            switch = True # (bal+fs)
            for blmethod_name, blmethod in blmethods[:1]:
                for fsmethod_name, fsmethod in fsmethods[:]:           
                    for percentile in percentiles[:]:
                        pct_name = f"Pct{percentile}"
                        for clmethod_name, clmethod in clmethods[:]:
                            name = "_".join([
                                cv_name,
                                pct_name,
                                fsmethod_name,
                                "---",
                                clmethod_name,
                                str(switch)
                            ])
                            # e.g. name
                            # metrics-Fold0_Pct10_Anova_---_CART_True
                            metrics = joblib.load(f"outsv2/outv2_8/{country}/metrics-{name}.jbl")
                            m["8_" + name] = metrics


        # Remove filterMethod- from filterMethod-Anova, etc. from the keys of m.
        mm = {}
        for k in m:
            if 'filterMethod-' in k:
                k0 = k.replace('filterMethod-', '')
                mm[k0] = m[k]
            else:
                mm[k] = m[k]
        m = mm


        def constructName(fold, percentile, fs, blc, clf, switch):
            """
            fs is either 
            "wrapper"
            "embedded"
            "Anova"
            "Info"

            blc is either
            None
            "SMOTE"
            "RandomOverSampler"
            "RandomUnderSampler"
            """
            name = ""
            name += "Fold" + str(fold) + "_"
            name += "Pct" + str(percentile) + "_"

            name += fs
            name += "_"

            if blc is None:
                name += "---" + "_"
            else:
                name += blc + "_"
            
            name += clf + "_"
            name += str(switch)

            if blc is None:
                if fs in ["PSO", "GA"]:
                    name = "8_" + name
                elif fs in ["LogisticRegressionWitL1Penalty", "RandomForest"]:
                    name = "6_" + name
                else:
                    name = "7_" + name
            else:
                if switch == False and fs in ["LogisticRegressionWitL1Penalty", "RandomForest"]:
                    name = "1_" + name
                elif switch == True and fs in ["LogisticRegressionWitL1Penalty", "RandomForest"]:
                    name = "2_" + name
                elif fs in ["Anova", "Info"]:
                    name = "3_" + name
                elif switch == False and fs in ["PSO", "GA"]:
                    name = "4_" + name
                elif switch == True and fs in ["PSO", "GA"]:
                    name = "5_" + name

            return name


        def descending_rank_scipy(d):
            values = list(d.values())
            
            # rankdata ranks in ascending order, so multiply by -1 for descending
            ranks = rankdata([-v for v in values], method='average')
            
            # Map ranks back to keys in original order
            return dict(zip(d.keys(), ranks))

        descending_rank_with_bisect = descending_rank_scipy



        # Find the cls that is most performant - the victor -
        # given fs, bal/switch
        vs = {}
        for fold, (train_idx, test_idx) in enumerate(joblib.load(f'cvs_{country}.jbl')['10 fold splits']): #10
            for percentile in percentiles:

                tmpd = {}
                for fs in ["Anova", "Info"] + ["LogisticRegressionWitL1Penalty", "RandomForest"] + ["PSO", "GA"]:
                    for blc in ["SMOTE", "RandomOverSampler", "RandomUnderSampler", None]:
                        if blc is None:
                            switches = [True]
                        else:
                            switches = [True, False]
                        for switch in switches:
                            for clf, clmethod in clmethods:

                                name = constructName(fold, percentile, fs, blc, clf, switch)
                                tmpd[name] = m[name][measure][0]

                vs[(fold, percentile)] = descending_rank_with_bisect(tmpd)
                        

        #
        # Calculate average ranks
        #

        # Compute the average for each pipeline
        d = defaultdict(float)
        for k in vs:
            v = vs[k]
            for k2 in v:
                name = "_".join(k2.split("_")[3:]) 
                d[name] += v[k2] * 1 / len(vs)

        # Rewrite pipeline name string into a dictionary comprising
        # the pipeline components and the corresponding average rank value.
        l = []
        values = []
        for k in d:
            fs, blc, clf, switch = k.split("_")
            val = d[k]
            values.append(val)
            
            l.append({
                'Classifier': clf,
                'Feature Selector': fs,
                'Balancing Combination': (blc, switch),
                'Average Rank': val
            })
  
        # Store the data in a dataframe
        df = pd.DataFrame(l)

        # Rename the axis labels
        mapping = {
            'CART': 'CART',
            'GradientBoostingClassifier': 'Gradient Boosting',
            'KNNClassifier': 'kNN',
            'LinearSVC': 'Linear SVC',
            'NaiveBayes': 'Naive Bayes',
            'RandomForestClassifier': 'Random Forest',
            'SVC': 'SV',
        }

        df["Classifier"] = df["Classifier"].replace(mapping)
        mapping = {
            'GA': 'GA',
            'Info': 'Info',
            'LogisticRegressionWitL1Penalty': 'Logistic Regression',
            'PSO': 'PSO',
            'RandomForest': 'Random Forest',
            'Anova': 'Anova',
        }
        df["Feature Selector"] = df["Feature Selector"].replace(mapping)

        mapping = {
            ('---', True): 'FS',
            ('RandomOverSampler', False): 'FS + ROS',
            ('RandomOverSampler', True): 'ROS + FS',
            ('RandomUnderSampler', False): 'FS + RUS',
            ('RandomUnderSampler', True): 'RUS + FS',
            ('SMOTE', False): 'FS + SMOTE',
            ('SMOTE', True): 'SMOTE + FS',
        }

        # Ensure second element is boolean (optional if already correct)
        def normalize_tuple(t):
            if isinstance(t[1], str):
                return (t[0], t[1] == 'True')
            return t

        # First, make sure that the y-coordinate value is Boolean.
        df["Balancing Combination"] = df["Balancing Combination"].apply(normalize_tuple)
        # Then, apply mapping to rename axis.
        df["Balancing Combination"] = df["Balancing Combination"].apply(lambda t: mapping.get(t, t))


        # Store in output_d
        output_d[country][measure] = df


        # Sort dataframe by averagee rank and round to 2 decimal places
        df_sorted = df.sort_values(by="Average Rank", ascending=True)
        df_sorted["Average Rank"] = df_sorted["Average Rank"].round(2)

        # Write to file in markdown the top 50 and bottom 50 average-ranked data.
        markdown_text = df_sorted.head(50).to_markdown(index=False)
        with open(f"./charts/{country}/averageranks_{measure}_top50.txt.md", "w") as f:
            f.write(markdown_text)
        markdown_text = df_sorted.tail(50).to_markdown(index=False)
        with open(f"./charts/{country}/averageranks_{measure}_bottom50.txt.md", "w") as f:
            f.write(markdown_text)


        # DOT PLOT

        # Find the cls that is most performant - the victor -
        # given fs, bal/switch
        vs_clf = {}
        for clf, clmethod in clmethods:
            vs_clf[clf] = {}   
            
            for fold, (train_idx, test_idx) in enumerate(joblib.load(f'cvs_{country}.jbl')['10 fold splits']): #10
                for percentile in percentiles:
                    for fs in ["Anova", "Info"] + ["LogisticRegressionWitL1Penalty", "RandomForest"] + ["PSO", "GA"]:
                        
                        tmpd = {}
            
                        for blc in ["SMOTE", "RandomOverSampler", "RandomUnderSampler", None]:
                            if blc is None:
                                switches = [True]
                            else:
                                switches = [True, False]
                            for switch in switches:
                            
                                name = constructName(fold, percentile, fs, blc, clf, switch)
                                tmpd[name] = m[name][measure][0]

                        vs_clf[clf][(fold, percentile, fs)] = descending_rank_with_bisect(tmpd)
                        

        # Calculate average rank
        d_clf = defaultdict(float)
        for clf in vs_clf:
            d_clf[clf] = defaultdict(float)
            v = vs_clf[clf] # e.g. vs_clf[clf][(fold, percentile, fs.replace("Anova", "t-Test"))]
            for triplet in v: # (fold, percentile, fs.replace("Anova", "t-Test"))
                # k looks like this: '3_Fold0_Pct10_t-Test_SMOTE_RandomForestClassifier_True'
                for k in v[triplet]:
                    name = "_".join([k.split("_")[4], "---", k.split("_")[6]]) # SMOTE_---_True
                    d_clf[clf][name] += v[triplet][k] * 1 / len(v)


        # Relabel axis
        d_clf2 = {}
        replace1 = {
            'RandomForestClassifier': 'Random Forest',
            'LinearSVC': 'Linear SV',
            'SVC': 'SV',
            'GradientBoostingClassifier': 'Gradient Boosting',
            'KNNClassifier': 'kNN',
            'NaiveBayes': 'Naive Bayes',
            'CART': 'CART'
        }
        replace2 = {
            'SMOTE_---_True': 'SMOTE+FS',
            'SMOTE_---_False': 'FS+SMOTE',
            'RandomOverSampler_---_True': 'ROS+FS',
            'RandomOverSampler_---_False': 'FS+ROS',
            'RandomUnderSampler_---_True': 'RUS+FS',
            'RandomUnderSampler_---_False': 'FS+RUS',
            '---_---_True': 'FS'   
        }
        for k in d_clf:
            d_clf2[replace1[k]] = {}
            for k2 in d_clf[k]:
                d_clf2[replace1[k]][replace2[k2]] = d_clf[k][k2]


        # Store data in a dataframe
        order = [
            'FS',
            'FS+ROS',
            'ROS+FS',
            'FS+RUS',
            'RUS+FS',
            'FS+SMOTE',
            'SMOTE+FS'
        ]
        df_dotplot = pd.DataFrame(d_clf2).round(2).reindex(order)
        dotplots[country][measure] = df_dotplot

        # Save to file the data in markdown
        markdown_text = df_dotplot.round(2).reindex(order).to_markdown(index=True)
        with open(f"./charts/{country}/averageranks_{measure}_{country}_clf-vs-blccombination.md", "w") as f:
            f.write(markdown_text)

        # Plotting in dot plots
        models = df_dotplot.columns
        strategies = df_dotplot.index
        colors = plt.cm.tab10(range(len(strategies)))

        plt.figure(figsize=(10, 5))

        for y, model in enumerate(models):
            for i, strategy in enumerate(strategies):
                plt.scatter(
                    df_dotplot.loc[strategy, model],
                    y,
                    color=colors[i],
                    s=60
                )

        plt.yticks(range(len(models)), models)
        plt.gca().invert_yaxis()   # top-down order
        plt.xlabel("Average Rank")

        # legend
        for i, strategy in enumerate(strategies):
            plt.scatter([], [], color=colors[i], label=strategy)

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"./charts/{country}/model_dotplot_{measure}_{country}.png", dpi=300, bbox_inches="tight")


joblib.dump(output_d, "./charts/output_d_averageRanks.jbl")
joblib.dump(dotplots, "./charts/dotplots.jbl")