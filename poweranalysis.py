from hierarch.power import DataSimulator
from hierarch.stats import hypothesis_test, confidence_interval
from hierarch.internal_functions import nb_data_grabber, GroupbyMean
import numpy as np
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
    t_test_pvals = []
    student_t_pvals = []

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

        """Calculate a p value with both types of t tests."""
        aggregator = GroupbyMean()
        test = aggregator.fit_transform(data)

        if compare == "means":
            sample_a, sample_b = nb_data_grabber(test, treatment_col, (1, 2))

            t_test_pvals.append(stats.ttest_ind(sample_a, sample_b, equal_var=False)[1])

            student_t_pvals.append(
                stats.ttest_ind(sample_a, sample_b, equal_var=True)[1]
            )
        elif compare == "corr":
            t_test_pvals.append(stats.pearsonr(test[:, treatment_col], test[:, -1])[1])

    if compare == "means":
        return perm_test_pvals, t_test_pvals, student_t_pvals
    elif compare == "corr":
        return perm_test_pvals, t_test_pvals


def conf_int_analysis(
    container,
    paramlist,
    true_slope=0,
    seed=None,
    interval=95,
    compare="corr",
    loops=1,
    treatment_col=0,
    bootstraps=1000,
    permutations="all",
    iterations=10,
    tolerance=0.5,
):

    """Seeded for reproducibility."""
    rng = np.random.default_rng(seed)

    """Initialize the DataSimulator with given parameters, random seed,
       then fit to the given data structure."""
    hierarchy = container.copy()
    sim = DataSimulator(paramlist, random_state=rng)
    sim.fit(hierarchy)

    coverage = 0
    w_coverage = 0
    t_coverage = 0

    for i in range(loops):

        """Generate a simulated dataset."""
        data = sim.generate()

        """Calculate a p value with hierarchical permutation."""
        lower, upper = confidence_interval(
            data,
            treatment_col,
            interval=interval,
            iterations=iterations,
            tolerance=tolerance,
            compare=compare,
            bootstraps=bootstraps,
            permutations=permutations,
            kind="bayesian",
            random_state=rng,
        )
        if lower <= true_slope <= upper:
            coverage += 1

        """Calculate intervals with both types of t tests."""
        aggregator = GroupbyMean()
        test = aggregator.fit_transform(data)

        alpha = (100 - interval) / 100

        if compare == "means":
            sample_a, sample_b = nb_data_grabber(test, treatment_col, (1, 2))
            sample_a = sms.DescrStatsW(sample_a)
            sample_b = sms.DescrStatsW(sample_b)

            cm = sms.CompareMeans(sample_b, sample_a)
            lower, upper = cm.tconfint_diff(alpha=alpha, usevar="unequal")
            if lower <= true_slope <= upper:
                w_coverage += 1

            lower, upper = cm.tconfint_diff(alpha=alpha, usevar="pooled")
            if lower <= true_slope <= upper:
                t_coverage += 1

        elif compare == "corr":
            X = test[:, treatment_col]
            X = sm.add_constant(X)

            mod = sm.OLS(test[:, -1], X)
            lower, upper = mod.fit().conf_int(alpha=alpha)[1]
            if lower <= true_slope <= upper:
                t_coverage += 1

    if compare == "means":
        return coverage, w_coverage, t_coverage

    elif compare == "corr":

        return coverage, t_coverage

