# coding: utf-8

import numpy as np

import itertools

from scipy import stats
from scipy.stats import probplot
from scipy.stats import chi2_contingency, wilcoxon, mannwhitneyu
from scipy.stats import ttest_1samp as tt1s, ttest_ind as ttind, ttest_rel as ttrel
from scipy.stats import pearsonr, spearmanr
from scipy.stats import kstest, ks_2samp, chisquare, fisher_exact

from statsmodels.stats.api import proportion_effectsize, NormalIndPower

from statsmodels.stats.weightstats import zconfint, CompareMeans, DescrStatsW
from statsmodels.stats.descriptivestats import sign_test

from statsmodels.stats.proportion import proportion_confint as prop_confint
from statsmodels.stats.proportion import samplesize_confint_proportion

from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.sandbox.stats.multicomp import multipletests

from collections import namedtuple


def mean_diff_confint_ind(sample1, sample2, alpha=0.05):
    """Доверительный интервал разности средних для двух независимых выборок

    Parameters
    ----------
    sample1 : array_like
        Первая выборка
    sample2 : array_like
        Вторая выборка
    alpha : float in (0, 1)
        Уровень доверия, рассчитывается как ``1-alpha``

    Returns
    -------
    lower, upper : floats
        Левая и правая граница доверительного интервала
    """
    cm = CompareMeans(DescrStatsW(sample1), DescrStatsW(sample2))
    return cm.tconfint_diff(alpha=alpha)


def mean_diff_confint_rel(sample1, sample2, alpha=0.05):
    """Доверительный интервал разности средних для двух связанных выборок

    Parameters
    ----------
    sample1 : array_like
        Первая выборка
    sample2 : array_like
        Вторая выборка
    alpha : float in (0, 1)
        Уровень доверия, рассчитывается как ``1-alpha``

    Returns
    -------
    lower, upper : floats
        Левая и правая граница доверительного интервала
    """
    return DescrStatsW(sample1 - sample2).tconfint_mean(alpha=alpha)


# Критерии Стьюдента

def ttest_1sample(sample, mean):
    """t-статистика для одной выборки.

    Двухсторонний тест для проверки равенства средного выборки sample и значения mean

    Parameters
    ----------
    sample : array_like
        Массив наблюдений
    mean : float or array_like
        Ожидаемое значение среднего

    Returns
    -------
    statistic : float or array
        t-statistic
    pvalue : float or array
        two-tailed p-value
    """
    return tt1s(a=sample, popmean=mean)


def ttest_ind(sample1, sample2):
    """t-статистика для двух независимых выборок.

    Двухсторонний тест для проверки равенства средних двух независимых выборок

    Parameters
    ----------
    sample1 : array_like
        Первая выборка
    sample1 : array_like
        Вторая выборка

    Returns
    -------
    statistic : float or array
        t-statistic
    pvalue : float or array
        two-tailed p-value
    """
    return ttind(sample1, sample2, equal_var=False)


def ttest_rel(sample1, sample2):
    """t-статистика для двух связанных выборок.

    Двухсторонний тест для проверки равенства средних двух связанных выборок

    Parameters
    ----------
    sample1 : array_like
        Первая выборка
    sample1 : array_like
        Вторая выборка

    Returns
    -------
    statistic : float or array
        t-statistic
    pvalue : float or array
        two-tailed p-value
    """
    return ttrel(sample1, sample2)

ShapiroResult = namedtuple('ShapiroResult', ('statistic', 'pvalue'))


def shapiro_test(sample):
    """Тест Шапиро-Уилка

    Проверяет гипотезу о нормальности распеределения

    Parameters
    ----------
    sample : array_like
        Массив наблюдений

    Returns
    -------
    statistic : float or array
        Статистика Шапиро-Уилка
    pvalue : float or array
        two-tailed p-value
    """
    res = stats.shapiro(sample)
    return ShapiroResult(*res)


def qq_plot(*args):
    """Квантиль-квантиль график, показывающий на сколько данные приближены к какому-либо распределению

    Parameters
    ----------
    sample : array_like
        Массив наблюдений

    Examples
    --------
    >>> _,_ = mrstat.qq_plot(sample, dist='norm', plot=plt)

    See Also
    --------
    scipy.stats.probplot
    """
    return probplot(*args)


# Непараметритические критерии для одной выборки
def z_confint(sample):
    """Доверительный интервал (95) основанный на нормальном распределении

    Parameters
    ----------
    sample : array_like
        Массив наблюдений

    Returns
    -------
    lower, upper : floats
        Левая и правая граница доверительного интервала
    """
    return zconfint(sample)

SigntestResult = namedtuple('SigntestResult', ('statistic', 'pvalue'))


def sign_test_1sample(sample, mean):
    """Одновыборочный критерий знаков

    Проверяет гипотезу о равенстве среднего конкретному значению mean

    Parameters
    ----------
    sample : array_like
        Массив наблюдений
    mean : float
        Проверяемое значение

    Returns
    -------
    statistic : float or array
        Статистика
    pvalue : float or array
        two-tailed p-value
    """
    res = sign_test(sample, mean)
    return SigntestResult(*res)


PermutationResult = namedtuple('PermutationResult', ('statistic', 'pvalue'))

def permutation_t_stat_1sample(sample, mean):
    """t-статистика для перестановочного критерия для одной выборки.

    Parameters
    ----------
    sample : array_like
        Массив наблюдений
    mean : float or array_like
        Ожидаемое значение среднего

    Returns
    -------
    statistic : float or array
        t-statistic
    """
    t_stat = sum(map(lambda x: x - mean, sample))
    return t_stat


def permutation_zero_distr_1sample(sample, mean, max_permutations=None):
    """Нулевое распределение для перестановочного критерия для одной выборки.

    Parameters
    ----------
    sample : array_like
        Массив наблюдений
    mean : float or array_like
        Ожидаемое значение среднего
    max_permutations : int or None
        Количество перестановок

    Returns
    -------
    distr : array_like
        Полученное распределение
    """
    centered_sample = map(lambda x: x - mean, sample)
    if max_permutations:
        signs_array = set([tuple(x) for x in 2 * np.random.randint(2, size=(max_permutations,
                                                                            len(sample))) - 1 ])
    else:
        signs_array = itertools.product([-1, 1], repeat=len(sample))
    distr = [sum(centered_sample * np.array(signs)) for signs in signs_array]
    return distr


def permutation_test_1sample(sample, mean, max_permutations=None, alternative='two-sided'):
    """Одновыборочный перестановочный критерий

    Проверяет гипотезу о равенстве среднего конкретному значению mean

    Parameters
    ----------
    sample : array_like
        Массив наблюдений
    mean : float
        Проверяемое значение
    max_permutations : int
        Количество производимых перестановок
    alternative : string in ['two-sided', 'less', 'greater']
        Тип проверяемой альтернативы

    Returns
    -------
    statistic : float or array
        Значение статистики
    pvalue : float or array
        p-value
    """
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    t_stat = permutation_t_stat_1sample(sample, mean)

    zero_distr = permutation_zero_distr_1sample(sample, mean, max_permutations)

    pvalue = 0.
    if alternative == 'two-sided':
        pvalue = sum([1. if abs(x) >= abs(t_stat) else 0. for x in zero_distr]) / len(zero_distr)

    if alternative == 'less':
        pvalue = sum([1. if x <= t_stat else 0. for x in zero_distr]) / len(zero_distr)

    if alternative == 'greater':
        pvalue = sum([1. if x >= t_stat else 0. for x in zero_distr]) / len(zero_distr)

    res = (t_stat, pvalue)
    return PermutationResult(*res)


# Непараметритические критерии для связных выборок
def sign_test_rel(sample1, sample2):
    """Двухвыборочный критерий знаков

    Проверяет гипотезу о равенстве средних двух связанных выборок

    Parameters
    ----------
    sample1 : array_like
        Первая выборка
    sample2 : array_like
        Вторая выборка

    Returns
    -------
    statistic : float or array
        Статистика
    pvalue : float or array
        two-tailed p-value
    """
    return sign_test_1sample(sample1 - sample2, 0.)


def wilcoxon_test_rel(sample1, sample2):
    """Двухвыборочный критерий знаковых рангов Вилкоксона

    Проверяет гипотезу об отсутствии смещения между распределениями

    Parameters
    ----------
    sample1 : array_like
        Первая выборка
    sample2 : array_like
        Вторая выборка

    Returns
    -------
    statistic : float or array
        Статистика
    pvalue : float or array
        p-value
    """
    return wilcoxon(sample1, sample2)


# Непараметрические притерии для несвязных выборок
def mannwhitneyu_test_ind(sample1, sample2, alternative='two-sided'):
    """U-критерий Манна — Уитни (англ. Mann — Whitney U-test) — статистический критерий,
    используемый для оценки различий между двумя независимыми выборками по уровню какого-либо признака,
    измеренного количественно.
    Позволяет выявлять различия в значении параметра между малыми выборками.

    Parameters
    ----------
    sample1 : array_like
        Первая выборка
    sample2 : array_like
        Вторая выборка
    alternative : string in ['two-sided', 'less', 'greater']
        Тип проверяемой альтернативы

    Returns
    -------
    statistic : float or array
        Значение статистики
    pvalue : float or array
        p-value
    """
    return mannwhitneyu(sample1, sample2, alternative=alternative)


def permutation_t_stat_ind(sample1, sample2):
    """t-статистика для перестановочного критерия для двух выборок выборки.

    Parameters
    ----------
    sample1 : array_like
        Первая выборка
    sample2 : array_like
        Вторая выборка

    Returns
    -------
    statistic : float or array
        t-statistic
    """
    return np.mean(sample1) - np.mean(sample2)


def get_random_combinations(n1, n2, max_permutations):
    """Вспомогателььный метод для получения случайных комбинаций

    Parameters
    ----------
    n1 : int
        Размер первой выборки
    n2 : int
        Размер второй выборки
    max_permutations : int
        Количество требуемых комбинаций

    Returns
    -------
    indexes : array
        Массив комбинаций индексов
    """
    index = range(n1 + n2)
    indices = set([tuple(index)])
    for i in range(max_permutations - 1):
        np.random.shuffle(index)
        indices.add(tuple(index))
    return [(index[:n1], index[n1:]) for index in indices]


def permutation_zero_dist_ind(sample1, sample2, max_permutations=None):
    """Нулевое распределение для перестановочного критерия для двух выборок.

    Parameters
    ----------
    sample1 : array_like
        Первая выборка
    sample2 : array_like
        Вторая выборка
    max_permutations : int or None
        Количество перестановок

    Returns
    -------
    distr : array_like
        Полученное распределение
    """
    joined_sample = np.hstack((sample1, sample2))
    n1 = len(sample1)
    n = len(joined_sample)

    if max_permutations:
        indices = get_random_combinations(n1, len(sample2), max_permutations)
    else:
        indices = [(list(index), filter(lambda i: i not in index, range(n))) for index in itertools.combinations(range(n), n1)]

    distr = [joined_sample[list(i[0])].mean() - joined_sample[list(i[1])].mean() for i in indices]
    return distr


def permutation_test_ind(sample1, sample2, max_permutations=None, alternative='two-sided'):
    """Двухвыборочный перестановочный критерий для независимых выборок

    Проверяет гипотезу о равенстве средних

    Parameters
    ----------
    sample1 : array_like
        Первая выборка
    sample2 : array_like
        Вторая выборка
    max_permutations : int
        Количество производимых перестановок
    alternative : string in ['two-sided', 'less', 'greater']
        Тип проверяемой альтернативы

    Returns
    -------
    statistic : float or array
        Значение статистики
    pvalue : float or array
        p-value
    """
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    t_stat = permutation_t_stat_ind(sample1, sample2)

    zero_distr = permutation_zero_dist_ind(sample1, sample2, max_permutations)

    pvalue = 0.
    if alternative == 'two-sided':
        pvalue = sum([1. if abs(x) >= abs(t_stat) else 0. for x in zero_distr]) / len(zero_distr)

    if alternative == 'less':
        pvalue = sum([1. if x <= t_stat else 0. for x in zero_distr]) / len(zero_distr)

    if alternative == 'greater':
        pvalue = sum([1. if x >= t_stat else 0. for x in zero_distr]) / len(zero_distr)

    res = (t_stat, pvalue)
    return PermutationResult(*res)


def permutation_test_rel(sample1, sample2, max_permutations=None, alternative='two-sided'):
    """Двухвыборочный перестановочный критерий для связанных выборок

    Проверяет гипотезу о равенстве средних

    Parameters
    ----------
    sample1 : array_like
        Первая выборка
    sample2 : array_like
        Вторая выборка
    max_permutations : int
        Количество производимых перестановок
    alternative : string in ['two-sided', 'less', 'greater']
        Тип проверяемой альтернативы

    Returns
    -------
    statistic : float or array
        Значение статистики
    pvalue : float or array
        p-value
    """
    return permutation_test_1sample(sample1-sample2, 0.,
                                    max_permutations=max_permutations, alternative=alternative)


# Bootstrap
def get_bootstrap_samples(data, n_samples=1000):
    """Получение случайной подвыборки (bootstrap)

    Parameters
    ----------
    data : array_like
        Базовая выборка
    n_samples : int
        Размер требуемой подвыборки

    Returns
    -------
    samples : array_like
        Случайная подвыборка
    """
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples


def bootstrap_conf_int(data, stat_func, alpha=0.05, n_samples=1000):
    """Доверительный интервал методом бутстрапа

    Parameters
    ----------
    data : array_like
        Проверяемая выборка
    stat_func : function
        Функция для рассчёта статистического критерия, например, np.mean или np.median
    alpha : float in (0, 1)
        Уровень доверия, рассчитывается как ``1-alpha``
    n_samples : int
        Размер подвыбрки

    Returns
    -------
    lower, upper : floats
        Левая и правая граница доверительного интервала

    Notes
    -----
    a = np.random.normal(size=1000)
    conf_int(a,np.median)
    """
    scores = [stat_func(sample) for sample in get_bootstrap_samples(data, n_samples)]
    return _bootstrap_get_confint(scores, alpha)


BootstrapResult = namedtuple('BootstrapResult', ('statistic', 'pvalue'))


def bootstrap_test(sample, mean, stat_func, n_samples=1000, alternative='two-sided'):
    """Одновыборочный bootstrap-критерий

    Parameters
    ----------
    sample : array_like
        Тестируемая выборка
    mean : float
        Проверяемое значение
    stat_func : function
        Функция для рассчёта статистического критерия, например, np.mean или np.median
    n_samples : int
        Размер подвыбрки для бутстрапа
    alternative : string
        Проверяемая альтернатива
         - 'two-sided' : H1: Значимое отклонение от p_0
         - 'greater' :   H1: Значимо больше p_0
         - 'less' :      H1: Значимо меньше p_0
    Returns
    -------
    statistic : float
        Значение статистики
    p_value : float
        Достигаемый уровень значимости
    """
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    param_d = [stat_func(i) for i in get_bootstrap_samples(sample, n_samples)]

    mean_p = np.mean(param_d)
    t_stat = stat_func(sample) - mean

    zero_dist = [(mm - mean_p) for mm in param_d]

    pvalue = 0
    if alternative == 'two-sided':
        pvalue = sum([1. if abs(x) >= abs(t_stat) else 0. for x in zero_dist]) / len(zero_dist)

    if alternative == 'less':
        pvalue = sum([1. if x <= t_stat else 0. for x in zero_dist]) / len(zero_dist)

    if alternative == 'greater':
        pvalue = sum([1. if x >= t_stat else 0. for x in zero_dist]) / len(zero_dist)

    res = (t_stat, pvalue)
    return BootstrapResult(*res)


def bootstrap_diff_conf_int(sample1, sample2, stat_func, alpha=0.05, n_samples=1000):
    """Двувыборочный bootstrap-критерий для независимых выборок

    Parameters
    ----------
    sample1 : array_like
        Тестируемая выборка
    sample2 : array_like
        Тестируемая выборка
    stat_func : function
        Функция для рассчёта статистического критерия, например, np.mean или np.median
    alpha : float in (0, 1)
        Уровень доверия, рассчитывается как ``1-alpha``
    n_samples : int
        Размер подвыбрки

    Returns
    -------
    lower, upper : floats
        Левая и правая граница доверительного интервала

    Notes
    -----
    a = np.random.normal(size=1000)
    b = np.random.normal(loc=2,size=1000)
    diff_conf_int(b,a,np.median)
    """
    scores_a = [stat_func(sample) for sample in get_bootstrap_samples(sample1, n_samples)]
    scores_b = [stat_func(sample) for sample in get_bootstrap_samples(sample2, n_samples)]
    delta_scores = [x[0] - x[1] for x in zip(scores_a, scores_b)]
    return _bootstrap_get_confint(delta_scores, alpha)


def _bootstrap_get_confint(data, alpha=0.05):
    """Получение границ интревала

    Parameters
    ----------
    data : array_like
        Исходный массив данных
    alpha : float in (0, 1)
        Уровень доверия, рассчитывается как ``1-alpha``

    Returns
    -------
    lower, upper : floats
        Левая и правая граница доверительного интервала
    """
    boundaries = np.percentile(data, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return tuple(boundaries)


# Критерии для доли
def proportions_confint(sample, alpha=0.05, method='wilson'):
    """Доверительный интервал для доли

    Parameters
    ----------
    sample : array_like
        Тестируемая выборка
    alpha : float in (0, 1)
        Уровень доверия, рассчитывается как ``1-alpha``
    method : string
        Метод рассчёта доверительного интервала
        поддерживаюся следующие варианты:

         - `normal` : asymptotic normal approximation
         - `agresti_coull` : Agresti-Coull interval
         - `beta` : Clopper-Pearson interval based on Beta distribution
         - `wilson` : Wilson Score interval
         - `jeffrey` : Jeffrey's Bayesian Interval
         - `binom_test` : experimental, inversion of binom_test
    Returns
    -------
    ci_low, ci_upp : float
        Левая и правая граница доверительного интервала
    """
    return prop_confint(count=sum(sample), nobs=sample.shape[0], alpha=alpha, method=method)


def proportions_z_stat_1sample(sample, p_0=0.5):
    """Одновыборочная Z-статистика для доли

    Parameters
    ----------
    sample : array_like
        Тестируемая выборка
    p_0 : array_like
        Проверяемое значение
    alternative : string
        Проверяемая альтернатива
         - 'two-sided' : H1: Значимое отклонение от p_0
         - 'greater' :   H1: Значимо больше p_0
         - 'less' :      H1: Значимо меньше p_0

    Returns
    -------
    z_stat : float
        Z-статистика
    """
    p = sample.mean()
    se = np.sqrt(p * (1 - p) / sample.shape[0])
    z_stat = (p - p_0) / se
    return z_stat

ProportionsResult = namedtuple('ProportionsResult', ('statistic', 'pvalue'))


def proportions_test_1sample(sample, p_0=0.5, alternative='two-sided'):
    """Одновыборочный критерий для доли

    Проверяет гипотезу о равенстве доли конкретному значению p_0

    Parameters
    ----------
    sample : array_like
        Массив наблюдений
    p_0 : float
        Проверяемое значение
    alternative : string in ['two-sided', 'less', 'greater']
        Тип проверяемой альтернативы

    Returns
    -------
    statistic : float or array
        Значение статистики
    pvalue : float or array
        p-value
    """
    z_stat = proportions_z_stat_1sample(sample=sample, p_0=p_0)
    pvalue = proportions_diff_z_test(z_stat=z_stat, alternative=alternative)
    res = (z_stat, pvalue)
    return ProportionsResult(*res)


def proportions_test_ind(sample1, sample2, alternative='two-sided'):
    """Двухвыборочный критерий для двух долей. Несвязвнные выборки

    Parameters
    ----------
    sample1 : array_like
        Первая выборка
    sample2 : array_like
        Вторая выборка
    alternative : string in ['two-sided', 'less', 'greater']
        Тип проверяемой альтернативы

    Returns
    -------
    statistic : float or array
        Значение статистики
    pvalue : float or array
        p-value
    """
    z_stat = proportions_diff_z_stat_ind(sample1, sample2)
    pvalue = proportions_diff_z_test(z_stat, alternative=alternative)
    res = (z_stat, pvalue)
    return ProportionsResult(*res)


def proportions_test_rel(sample1, sample2, alternative='two-sided'):
    """Двухвыборочный критерий для двух долей. Несвязвнные выборки

    Parameters
    ----------
    sample1 : array_like
        Первая выборка
    sample2 : array_like
        Вторая выборка
    alternative : string in ['two-sided', 'less', 'greater']
        Тип проверяемой альтернативы

    Returns
    -------
    statistic : float or array
        Значение статистики
    pvalue : float or array
        p-value
    """
    z_stat = proportions_diff_z_stat_rel(sample1, sample2)
    pvalue = proportions_diff_z_test(z_stat, alternative=alternative)
    res = (z_stat, pvalue)
    return ProportionsResult(*res)


def proportions_diff_confint_ind(sample1, sample2, alpha=0.05):
    """Доверительный интервал для разности долей (независимые выборки)

    Parameters
    ----------
    sample1 : array_like
        Тестируемая выборка
    sample2 : array_like
        Тестируемая выборка
    alpha : float in (0, 1)
        Уровень доверия, рассчитывается как ``1-alpha``

    Returns
    -------
    lower, upper : floats
        Левая и правая граница доверительного интервала
    """
    z = stats.norm.ppf(1 - alpha / 2.)

    p1 = float(sum(sample1)) / len(sample1)
    p2 = float(sum(sample2)) / len(sample2)

    lower = (p1 - p2) - z * np.sqrt(p1 * (1 - p1)/ len(sample1) + p2 * (1 - p2)/ len(sample2))
    upper = (p1 - p2) + z * np.sqrt(p1 * (1 - p1)/ len(sample1) + p2 * (1 - p2)/ len(sample2))

    return lower, upper


def proportions_diff_z_stat_ind(sample1, sample2):
    """Z-статистика для разности двух долей (независимые выборки)

    Parameters
    ----------
    sample1 : array_like
        Тестируемая выборка
    sample2 : array_like
        Тестируемая выборка

    Returns
    -------
    z : floats
        Значение Z-статистики
    """
    n1 = len(sample1)
    n2 = len(sample2)

    p1 = float(sum(sample1)) / n1
    p2 = float(sum(sample2)) / n2
    p = float(p1 * n1 + p2 * n2) / (n1 + n2)

    return (p1 - p2) / np.sqrt(p * (1 - p) * (1. / n1 + 1. / n2))


def proportions_confint_diff_rel(sample1, sample2, alpha=0.05):
    """Доверительный интервал для разности долей (связанные выборки)

    Parameters
    ----------
    sample1 : array_like
        Тестируемая выборка
    sample2 : array_like
        Тестируемая выборка
    alpha : float in (0, 1)
        Уровень доверия, рассчитывается как ``1-alpha``

    Returns
    -------
    lower, upper : floats
        Левая и правая граница доверительного интервала
    """
    z = stats.norm.ppf(1 - alpha / 2.)
    sample = zip(sample1, sample2)
    n = len(sample)

    f = sum([1 if (x[0] == 1 and x[1] == 0) else 0 for x in sample])
    g = sum([1 if (x[0] == 0 and x[1] == 1) else 0 for x in sample])

    lower = float(f - g) / n - z * np.sqrt(float((f + g)) / n ** 2 - float((f - g) ** 2) / n ** 3)
    upper = float(f - g) / n + z * np.sqrt(float((f + g)) / n ** 2 - float((f - g) ** 2) / n ** 3)
    return lower, upper


def proportions_diff_z_stat_rel(sample1, sample2):
    """Z-статистика для разности двух долей (связанные выборки)

    Parameters
    ----------
    sample1 : array_like
        Тестируемая выборка
    sample2 : array_like
        Тестируемая выборка

    Returns
    -------
    z : floats
        Значение Z-статистики
    """
    sample = zip(sample1, sample2)
    n = len(sample)

    f = sum([1 if (x[0] == 1 and x[1] == 0) else 0 for x in sample])
    g = sum([1 if (x[0] == 0 and x[1] == 1) else 0 for x in sample])

    return float(f - g) / np.sqrt(f + g - float((f - g) ** 2) / n)


def proportions_diff_z_test(z_stat, alternative='two-sided'):
    """Уровень значимости для доли при нормальном распределении

    Parameters
    ----------
    z_stat : float
        Значение Z-статистики
    alternative : string
        Проверяемая альтернатива
         - 'two-sided'_0
         - 'greater'
         - 'less'
    Returns
    -------
    p_value : float
        Достигаемый уровень значимости

    """
    import scipy
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    if alternative == 'two-sided':
        return 2 * (1 - scipy.stats.norm.cdf(np.abs(z_stat)))

    if alternative == 'less':
        return scipy.stats.norm.cdf(z_stat)

    if alternative == 'greater':
        return 1 - scipy.stats.norm.cdf(z_stat)


def proportions_confint_samplesize(proportion, half_length=0.05, alpha=0.05):
    """Размер выборки для доли для получения требуемого доверительного интервала

    Parameters
    ----------
    proportion : float in (0, 1)
        доля, например, 0.5
    half_length : float in (0, 1)
        половина требуемой длинны доверительного интервала
    alpha : float in (0, 1)
        уровень доверия, по-умолчанию 0.05,
        покрывает двухсторонний интервал ``1 - alpha``

    Returns
    -------
    n : float
        Количество наблюдений для получения требуемого доверительного интервала
    """
    return samplesize_confint_proportion(proportion=proportion, half_length=half_length, alpha=alpha)


def proportions_two_samplesize(p1, p2, frac=0.5, power=0.8, alpha=0.05):
    """Размер выборки для двух долей

    Parameters
    ----------
    p1 : float in (0, 1)
        Улучшаемый показатель, например, 0.1
    p2 : float in (0, 1)
        Требуемое значение показателя, например, 0.1 * 1.2 (на 20 процентов больше p1)
    frac : float in (0, 1)
        Пропорция контрольной и общего размера теста, например, 0.2 - это 20% от всего эксперимента
    power : float in (0, 1)
        Мощность, по умолчанию 0.8
    alpha : float in (0, 1)
        Достигаемый уровень значимсоти, по-умолчанию 0.05

    Returns
    -------
    n1, n2 : float
        Необходимое количество наблюдений в контрольной и тестовой группах,
        сумма показывает общее количество необходимых наблюдений

    Notes
    -----
    Используется для вычисления размера требуемой выборки при проведении AB-теста
    p1 - показатель который необходимо улучшить до уровня p2.
    В примере:
        p1 - 0.1 (10%)
        p2 - 0.1*1.2 - требуется улучшить на 20%
        frac - 0.2 - 20% контроль, 80% тест

    Examples
    --------
    >>> proportions_two_samplesize(0.1, 0.1 * 1.2, frac=0.2)
    (2396.5, 9586.0)
    """
    ratio = frac / (1. - frac)
    es = proportion_effectsize(p1, p2)
    n = np.floor(NormalIndPower().solve_power(es, power=power, alpha=alpha, ratio=ratio))
    n1, n2 = n * ratio, n
    return n1, n2


# Корреляции
PearsonrResult = namedtuple('PearsonrResult', ('correlation', 'pvalue'))


def pearson_test(sample1, sample2):
    """Коэффициент корреляции Пирсона для двух выборок

    Parameters
    ----------
    sample1 : array_like
        Тестируемая выборка
    sample2 : array_like
        Тестируемая выборка

    Returns
    -------
    correlation : floats
        Значение коэффициента корреялции
    pvalue : float
        Достигаемый уровень значимости
    """
    res = pearsonr(sample1, sample2)
    return PearsonrResult(*res)


def spearman_test(sample1, sample2):
    """Коэффициент корреляции Спирмана для двух выборок

    Parameters
    ----------
    sample1 : array_like
        Тестируемая выборка
    sample2 : array_like
        Тестируемая выборка

    Returns
    -------
    correlation : floats
        Значение коэффициента корреялции
    pvalue : float
        Достигаемый уровень значимости
    """
    return spearmanr(sample1, sample2)

