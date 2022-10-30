#![allow(missing_docs)]

use std::mem;

#[cfg(test)]
mod tests;

fn local_sort(v: &mut [f64]) {
    v.sort_by(|x: &f64, y: &f64| x.total_cmp(y));
}

/// Trait that provides simple descriptive statistics on a univariate set of numeric samples.
pub trait Stats {
    /// Sum of the samples.
    ///
    /// Note: this method sacrifices performance at the altar of accuracy
    /// Depends on IEEE 754 arithmetic guarantees. See proof of correctness at:
    /// ["Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric
    /// Predicates"][paper]
    ///
    /// [paper]: https://www.cs.cmu.edu/~quake-papers/robust-arithmetic.ps
    fn sum(&self) -> f64;

    /// Minimum value of the samples.
    fn min(&self) -> f64;

    /// Maximum value of the samples.
    fn max(&self) -> f64;

    /// Arithmetic mean (average) of the samples: sum divided by sample-count.
    ///
    /// See: <https://en.wikipedia.org/wiki/Arithmetic_mean>
    fn mean(&self) -> f64;

    /// Median of the samples: value separating the lower half of the samples from the higher half.
    /// Equal to `self.percentile(50.0)`.
    ///
    /// See: <https://en.wikipedia.org/wiki/Median>
    fn median(&self) -> f64;

    /// Variance of the samples: bias-corrected mean of the squares of the differences of each
    /// sample from the sample mean. Note that this calculates the _sample variance_ rather than the
    /// population variance, which is assumed to be unknown. It therefore corrects the `(n-1)/n`
    /// bias that would appear if we calculated a population variance, by dividing by `(n-1)` rather
    /// than `n`.
    ///
    /// See: <https://en.wikipedia.org/wiki/Variance>
    fn var(&self) -> f64;

    /// Standard deviation: the square root of the sample variance.
    ///
    /// Note: this is not a robust statistic for non-normal distributions. Prefer the
    /// `median_abs_dev` for unknown distributions.
    ///
    /// See: <https://en.wikipedia.org/wiki/Standard_deviation>
    fn std_dev(&self) -> f64;

    /// Standard deviation as a percent of the mean value. See `std_dev` and `mean`.
    ///
    /// Note: this is not a robust statistic for non-normal distributions. Prefer the
    /// `median_abs_dev_pct` for unknown distributions.
    fn std_dev_pct(&self) -> f64;

    /// Scaled median of the absolute deviations of each sample from the sample median. This is a
    /// robust (distribution-agnostic) estimator of sample variability. Use this in preference to
    /// `std_dev` if you cannot assume your sample is normally distributed. Note that this is scaled
    /// by the constant `1.4826` to allow its use as a consistent estimator for the standard
    /// deviation.
    ///
    /// See: <https://en.wikipedia.org/wiki/Median_absolute_deviation>
    fn median_abs_dev(&self) -> f64;

    /// Median absolute deviation as a percent of the median. See `median_abs_dev` and `median`.
    fn median_abs_dev_pct(&self) -> f64;

    /// Percentile: the value below which `pct` percent of the values in `self` fall. For example,
    /// percentile(95.0) will return the value `v` such that 95% of the samples `s` in `self`
    /// satisfy `s <= v`.
    ///
    /// Calculated by linear interpolation between closest ranks.
    ///
    /// See: <https://en.wikipedia.org/wiki/Percentile>
    fn percentile(&self, pct: f64) -> f64;

    /// Quartiles of the sample: three values that divide the sample into four equal groups, each
    /// with 1/4 of the data. The middle value is the median. See `median` and `percentile`. This
    /// function may calculate the 3 quartiles more efficiently than 3 calls to `percentile`, but
    /// is otherwise equivalent.
    ///
    /// See also: <https://en.wikipedia.org/wiki/Quartile>
    fn quartiles(&self) -> (f64, f64, f64);

    /// Inter-quartile range: the difference between the 25th percentile (1st quartile) and the 75th
    /// percentile (3rd quartile). See `quartiles`.
    ///
    /// See also: <https://en.wikipedia.org/wiki/Interquartile_range>
    fn iqr(&self) -> f64;
}

/// Extracted collection of all the summary statistics of a sample set.
#[derive(Debug, Clone, PartialEq, Copy)]
#[allow(missing_docs)]
pub struct Summary {
    pub sum: f64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub var: f64,
    pub std_dev: f64,
    pub std_dev_pct: f64,
    pub median_abs_dev: f64,
    pub median_abs_dev_pct: f64,
    pub quartiles: (f64, f64, f64),
    pub iqr: f64,
}

impl Summary {
    /// Construct a new summary of a sample set.
    pub fn new(samples: &[f64]) -> Summary {
        Summary {
            sum: samples.sum(),
            min: samples.min(),
            max: samples.max(),
            mean: samples.mean(),
            median: samples.median(),
            var: samples.var(),
            std_dev: samples.std_dev(),
            std_dev_pct: samples.std_dev_pct(),
            median_abs_dev: samples.median_abs_dev(),
            median_abs_dev_pct: samples.median_abs_dev_pct(),
            quartiles: samples.quartiles(),
            iqr: samples.iqr(),
        }
    }
}

impl Stats for [f64] {
    // FIXME #11059 handle NaN, inf and overflow
    fn sum(&self) -> f64 {
        let mut partials = vec![];

        for &x in self {
            let mut x = x;
            let mut j = 0;
            // This inner loop applies `hi`/`lo` summation to each
            // partial so that the list of partial sums remains exact.
            for i in 0..partials.len() {
                let mut y: f64 = partials[i];
                if x.abs() < y.abs() {
                    mem::swap(&mut x, &mut y);
                }
                // Rounded `x+y` is stored in `hi` with round-off stored in
                // `lo`. Together `hi+lo` are exactly equal to `x+y`.
                let hi = x + y;
                let lo = y - (hi - x);
                if lo != 0.0 {
                    partials[j] = lo;
                    j += 1;
                }
                x = hi;
            }
            if j >= partials.len() {
                partials.push(x);
            } else {
                partials[j] = x;
                partials.truncate(j + 1);
            }
        }
        let zero: f64 = 0.0;
        partials.iter().fold(zero, |p, q| p + *q)
    }

    fn min(&self) -> f64 {
        assert!(!self.is_empty());
        self.iter().fold(self[0], |p, q| p.min(*q))
    }

    fn max(&self) -> f64 {
        assert!(!self.is_empty());
        self.iter().fold(self[0], |p, q| p.max(*q))
    }

    fn mean(&self) -> f64 {
        assert!(!self.is_empty());
        self.sum() / (self.len() as f64)
    }

    fn median(&self) -> f64 {
        self.percentile(50_f64)
    }

    fn var(&self) -> f64 {
        if self.len() < 2 {
            0.0
        } else {
            let mean = self.mean();
            let mut v: f64 = 0.0;
            for s in self {
                let x = *s - mean;
                v += x * x;
            }
            // N.B., this is _supposed to be_ len-1, not len. If you
            // change it back to len, you will be calculating a
            // population variance, not a sample variance.
            let denom = (self.len() - 1) as f64;
            v / denom
        }
    }

    fn std_dev(&self) -> f64 {
        self.var().sqrt()
    }

    fn std_dev_pct(&self) -> f64 {
        let hundred = 100_f64;
        (self.std_dev() / self.mean()) * hundred
    }

    fn median_abs_dev(&self) -> f64 {
        let med = self.median();
        let abs_devs: Vec<f64> = self.iter().map(|&v| (med - v).abs()).collect();
        // This constant is derived by smarter statistics brains than me, but it is
        // consistent with how R and other packages treat the MAD.
        let number = 1.4826;
        abs_devs.median() * number
    }

    fn median_abs_dev_pct(&self) -> f64 {
        let hundred = 100_f64;
        (self.median_abs_dev() / self.median()) * hundred
    }

    fn percentile(&self, pct: f64) -> f64 {
        let mut tmp = self.to_vec();
        local_sort(&mut tmp);
        percentile_of_sorted(&tmp, pct)
    }

    fn quartiles(&self) -> (f64, f64, f64) {
        let mut tmp = self.to_vec();
        local_sort(&mut tmp);
        let first = 25_f64;
        let a = percentile_of_sorted(&tmp, first);
        let second = 50_f64;
        let b = percentile_of_sorted(&tmp, second);
        let third = 75_f64;
        let c = percentile_of_sorted(&tmp, third);
        (a, b, c)
    }

    fn iqr(&self) -> f64 {
        let (a, _, c) = self.quartiles();
        c - a
    }
}

// Helper function: extract a value representing the `pct` percentile of a sorted sample-set, using
// linear interpolation. If samples are not sorted, return nonsensical value.
fn percentile_of_sorted(sorted_samples: &[f64], pct: f64) -> f64 {
    assert!(!sorted_samples.is_empty());
    if sorted_samples.len() == 1 {
        return sorted_samples[0];
    }
    let zero: f64 = 0.0;
    assert!(zero <= pct);
    let hundred = 100_f64;
    assert!(pct <= hundred);
    if pct == hundred {
        return sorted_samples[sorted_samples.len() - 1];
    }
    let length = (sorted_samples.len() - 1) as f64;
    let rank = (pct / hundred) * length;
    let lrank = rank.floor();
    let d = rank - lrank;
    let n = lrank as usize;
    let lo = sorted_samples[n];
    let hi = sorted_samples[n + 1];
    lo + (hi - lo) * d
}

/// Winsorize a set of samples, replacing values above the `100-pct` percentile
/// and below the `pct` percentile with those percentiles themselves. This is a
/// way of minimizing the effect of outliers, at the cost of biasing the sample.
/// It differs from trimming in that it does not change the number of samples,
/// just changes the values of those that are outliers.
///
/// See: <https://en.wikipedia.org/wiki/Winsorising>
pub fn winsorize(samples: &mut [f64], pct: f64) {
    let mut tmp = samples.to_vec();
    local_sort(&mut tmp);
    let lo = percentile_of_sorted(&tmp, pct);
    let hundred = 100_f64;
    let hi = percentile_of_sorted(&tmp, hundred - pct);
    for samp in samples {
        if *samp > hi {
            *samp = hi
        } else if *samp < lo {
            *samp = lo
        }
    }
}
