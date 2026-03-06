import joblib
import time
from scipy.stats import dirichlet
from pathlib import Path


# sum(datasets['X_russian']['I3']) is 9, almost all entries are 0; I3 is unreliable supplier
# taiwanese data has a constant column - this triggers a warning, which is harmless
datasets = joblib.load('datasets.jbl')

countries = ["polish", "russian", "taiwanese"]

for country in countries:
    X = datasets[f'X_{country}']
    y = datasets[f'y_{country}'].squeeze() # converts 1d dataframe to series
    
    if country == "polish":
        mask = X["year"] == 5.0
        X = X[mask].drop(columns=["year"])
        y = y[mask]


def chkcond(methodX, methodY):
    """
    This function helps to pick out the appropriate pairs of pipelines
    X and Y for comparison.

    We want to compare a pipeline that has a resampler,
    with one configured in the same way but without the resampler.

    And we want to compare a pipeline that has a resampler,
    with the pipeline obtained by switching the order of
    the feature selector and the resampler.
    """
    # 't-Test_SMOTE_RandomForestClassifier_True'
    split_methodX = methodX.split('_')
    split_methodY = methodY.split('_')

    if split_methodX[0] == 'Correlation' or split_methodY[0] == 'Correlation':
        return False

    if all([
        split_methodX[-2] == 'GradientBoostingClassifier',
        split_methodY[-2] == 'GradientBoostingClassifier',
    ]):
        if any([
            all([
                split_methodX[1] != '---',
                split_methodY[1] == '---',
                split_methodX[0] == split_methodY[0],
            ]),
            all([
                split_methodX[-1] == 'True',
                split_methodY[-1] == 'False',
                split_methodX[1] != '---',
                split_methodY[1] != '---',
                split_methodX[0] == split_methodY[0],
                split_methodX[1] == split_methodY[1],
                #split_methodX[2] == split_methodY[2],
            ]),
        ]):
            return True
    return False


for measure in ['roc_aucs', 'f1_scores', 'g_means']:
    for country in countries:
        # results contain the average ranks computed prior to the current step
        results = joblib.load(f"./charts/results_{measure}_{country}.jbl")
        genlist = list(results.keys())

        for methodX in genlist[:len(genlist)]:
            for methodY in genlist[:len(genlist)]:

                comparetext = methodX + "___" + methodY
                #outfilename = f"./charts/mc/results_{measure}_{country}_{comparetext}.jbl"
                # The following is better because file names too long, Windows can't handle.
                outfilename = fr"C:\temp\results_{measure}_{country}_{comparetext}.jbl"

                if chkcond(methodX, methodY):
                    # The rankings of the method through the many tests
                    VarX = dict(results[methodX])
                    VarY = dict(results[methodY])
                    VarZ = {}

                    for key in VarX:
                        VarZ[key] = VarX[key] - VarY[key]

                    keys = VarX.keys()

                    VarX2 = []
                    VarY2 = []
                    Z = [] # to be conformant later
                    for key in keys:
                        VarX2.append(VarX[key])
                        VarY2.append(VarY[key])
                        Z.append(VarX[key] - VarY[key])

                    z0 = 0
                    s = 0.5
                    q = len(Z)
                    r = 0.001
                    Z = [z0] + Z

                    # Define alpha
                    alpha = [s] + q * [1]

                    # Create Dirichlet distribution object
                    dist = dirichlet(alpha)

                    # Generate samples
                    seed = 42
                    # based on the example from page 23 of the paper
                    samples = dist.rvs(size=150000, random_state=seed) 

                    # Compute the 3 Bayesian posterior probabilities
                    thetas = []

                    for idx, row in enumerate(samples[:]):
                        if idx % 1000 == 0:
                            print(comparetext, idx)
                        
                        tmpl = 0
                        tmpe = 0
                        tmpr = 0
                        for i in range(q + 1):
                            for j in range(q + 1):
                                # case of Y is better than X
                                if Z[i] + Z[j] < - 2 * r:
                                    tmpl += row[i] * row[j]

                                # case of Y is better than X
                                if Z[i] + Z[j] > 2 * r:
                                    tmpr += row[i] * row[j]            

                                # case of both methods are practically equivalent
                                if Z[i] + Z[j] >= - 2 * r and Z[i] + Z[j] <= 2 * r: 
                                    tmpe += row[i] * row[j]
                        
                        # Z<0, Z>0
                        # <=> X<Y, X>Y 
                        # <=> X better ranked than Y, Y better ranked than X
                        # This is because, the lower the rank, the better.
                        thetas.append((
                            tmpl, tmpe, tmpr    
                        ))

                    out = {
                        "samples": samples,
                        "thetas": thetas,
                    }

                    joblib.dump(out, outfilename)

