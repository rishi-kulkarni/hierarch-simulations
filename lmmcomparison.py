from hierarch.power import DataSimulator
from hierarch.stats import hypothesis_test, confidence_interval
from hierarch.internal_functions import nb_data_grabber, GroupbyMean
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.api as sms
import statsmodels.api as sm


def poweranalysis(
    container,
    paramlist,
    seed=None,
    compare="means",
    loops=1,
    treatment_col=0,
    bootstraps=1000,
    permutations="all",
):

    """Seeded for reproducibility."""
    rng = np.random.default_rng(seed)

    """Initialize the DataSimulator with given parameters, random seed,
       then fit to the given data structure."""
    hierarchy = container.copy()
    sim = DataSimulator(paramlist, random_state=rng)
    sim.fit(hierarchy)

    perm_test_pvals = []
    lmm_pvals = []

    for i in range(loops):

        """Generate a simulated dataset."""
        data = sim.generate()

        """Calculate a p value with hierarchical permutation."""
        perm_pval = hypothesis_test(
            data,
            treatment_col=treatment_col,
            bootstraps=bootstraps,
            permutations=permutations,
            kind="weights",
            random_state=rng,
        )
        perm_test_pvals.append(perm_pval)

        """Calculate a p value with a Linear Mixed Model. Presumes 2 levels."""
        col_names = [f"Level_{idx}" for idx, v in enumerate(container)]
        col_names.append("y")
        df = pd.DataFrame(data=data, columns=col_names)

        model = sm.MixedLM.from_formula(
            # Level_0 is the "treatment"
            "y ~ Level_0",
            # see https://www.statsmodels.org/devel/examples/notebooks/generated/variance_components.html
            # this is more or less equivalent to "y ~ Level_0 + (1 | Level_1)"
            # in Wilkinson notation
            groups="Level_1",
            data=df,
        )
        result = model.fit()
        # by specifying our model like this, y is a function of the intercept, so
        # that's the p-value we care about
        lmm_pvals.append(result.pvalues.Level_0)

    return perm_test_pvals, lmm_pvals
