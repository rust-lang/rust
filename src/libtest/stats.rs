// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(missing_doc)]

use std::collections::hashmap;
use std::fmt::Show;
use std::hash::Hash;
use std::io;
use std::mem;
use std::num::Zero;
use std::num;

fn local_cmp<T:Float>(x: T, y: T) -> Ordering {
    // arbitrarily decide that NaNs are larger than everything.
    if y.is_nan() {
        Less
    } else if x.is_nan() {
        Greater
    } else if x < y {
        Less
    } else if x == y {
        Equal
    } else {
        Greater
    }
}

fn local_sort<T: Float>(v: &mut [T]) {
    v.sort_by(|x: &T, y: &T| local_cmp(*x, *y));
}

/// Trait that provides simple descriptive statistics on a univariate set of numeric samples.
pub trait Stats <T: FloatMath + FromPrimitive>{

    /// Sum of the samples.
    ///
    /// Note: this method sacrifices performance at the altar of accuracy
    /// Depends on IEEE-754 arithmetic guarantees. See proof of correctness at:
    /// ["Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates"]
    /// (http://www.cs.cmu.edu/~quake-papers/robust-arithmetic.ps)
    /// *Discrete & Computational Geometry 18*, 3 (Oct 1997), 305-363, Shewchuk J.R.
    fn sum(self) -> T;

    /// Minimum value of the samples.
    fn min(self) -> T;

    /// Maximum value of the samples.
    fn max(self) -> T;

    /// Arithmetic mean (average) of the samples: sum divided by sample-count.
    ///
    /// See: https://en.wikipedia.org/wiki/Arithmetic_mean
    fn mean(self) -> T;

    /// Median of the samples: value separating the lower half of the samples from the higher half.
    /// Equal to `self.percentile(50.0)`.
    ///
    /// See: https://en.wikipedia.org/wiki/Median
    fn median(self) -> T;

    /// Variance of the samples: bias-corrected mean of the squares of the differences of each
    /// sample from the sample mean. Note that this calculates the _sample variance_ rather than the
    /// population variance, which is assumed to be unknown. It therefore corrects the `(n-1)/n`
    /// bias that would appear if we calculated a population variance, by dividing by `(n-1)` rather
    /// than `n`.
    ///
    /// See: https://en.wikipedia.org/wiki/Variance
    fn var(self) -> T;

    /// Standard deviation: the square root of the sample variance.
    ///
    /// Note: this is not a robust statistic for non-normal distributions. Prefer the
    /// `median_abs_dev` for unknown distributions.
    ///
    /// See: https://en.wikipedia.org/wiki/Standard_deviation
    fn std_dev(self) -> T;

    /// Standard deviation as a percent of the mean value. See `std_dev` and `mean`.
    ///
    /// Note: this is not a robust statistic for non-normal distributions. Prefer the
    /// `median_abs_dev_pct` for unknown distributions.
    fn std_dev_pct(self) -> T;

    /// Scaled median of the absolute deviations of each sample from the sample median. This is a
    /// robust (distribution-agnostic) estimator of sample variability. Use this in preference to
    /// `std_dev` if you cannot assume your sample is normally distributed. Note that this is scaled
    /// by the constant `1.4826` to allow its use as a consistent estimator for the standard
    /// deviation.
    ///
    /// See: http://en.wikipedia.org/wiki/Median_absolute_deviation
    fn median_abs_dev(self) -> T;

    /// Median absolute deviation as a percent of the median. See `median_abs_dev` and `median`.
    fn median_abs_dev_pct(self) -> T;

    /// Percentile: the value below which `pct` percent of the values in `self` fall. For example,
    /// percentile(95.0) will return the value `v` such that 95% of the samples `s` in `self`
    /// satisfy `s <= v`.
    ///
    /// Calculated by linear interpolation between closest ranks.
    ///
    /// See: http://en.wikipedia.org/wiki/Percentile
    fn percentile(self, pct: T) -> T;

    /// Quartiles of the sample: three values that divide the sample into four equal groups, each
    /// with 1/4 of the data. The middle value is the median. See `median` and `percentile`. This
    /// function may calculate the 3 quartiles more efficiently than 3 calls to `percentile`, but
    /// is otherwise equivalent.
    ///
    /// See also: https://en.wikipedia.org/wiki/Quartile
    fn quartiles(self) -> (T,T,T);

    /// Inter-quartile range: the difference between the 25th percentile (1st quartile) and the 75th
    /// percentile (3rd quartile). See `quartiles`.
    ///
    /// See also: https://en.wikipedia.org/wiki/Interquartile_range
    fn iqr(self) -> T;
}

/// Extracted collection of all the summary statistics of a sample set.
#[deriving(Clone, PartialEq)]
#[allow(missing_doc)]
pub struct Summary<T> {
    pub sum: T,
    pub min: T,
    pub max: T,
    pub mean: T,
    pub median: T,
    pub var: T,
    pub std_dev: T,
    pub std_dev_pct: T,
    pub median_abs_dev: T,
    pub median_abs_dev_pct: T,
    pub quartiles: (T,T,T),
    pub iqr: T,
}

impl<T: FloatMath + FromPrimitive> Summary<T> {

    /// Construct a new summary of a sample set.
    pub fn new(samples: &[T]) -> Summary<T> {
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
            iqr: samples.iqr()
        }
    }
}

impl<'a, T: FloatMath + FromPrimitive> Stats<T> for &'a [T] {

    // FIXME #11059 handle NaN, inf and overflow
    fn sum(self) -> T {
        let mut partials = vec![];

        for &mut x in self.iter() {
            let mut j = 0;
            // This inner loop applies `hi`/`lo` summation to each
            // partial so that the list of partial sums remains exact.
            for i in range(0, partials.len()) {
                let mut y = partials[i];
                if num::abs(x) < num::abs(y) {
                    mem::swap(&mut x, &mut y);
                }
                // Rounded `x+y` is stored in `hi` with round-off stored in
                // `lo`. Together `hi+lo` are exactly equal to `x+y`.
                let hi = x + y;
                let lo = y - (hi - x);
                if !lo.is_zero() {
                    *partials.get_mut(j) = lo;
                    j += 1;
                }
                x = hi;
            }
            if j >= partials.len() {
                partials.push(x);
            } else {
                *partials.get_mut(j) = x;
                partials.truncate(j+1);
            }
        }
        let zero: T = Zero::zero();
        partials.iter().fold(zero, |p, q| p + *q)
    }

    fn min(self) -> T {
        assert!(self.len() != 0);
        self.iter().fold(self[0], |p, q| p.min(*q))
    }

    fn max(self) -> T {
        assert!(self.len() != 0);
        self.iter().fold(self[0], |p, q| p.max(*q))
    }

    fn mean(self) -> T {
        assert!(self.len() != 0);
        self.sum() / FromPrimitive::from_uint(self.len()).unwrap()
    }

    fn median(self) -> T {
        self.percentile(FromPrimitive::from_uint(50).unwrap())
    }

    fn var(self) -> T {
        if self.len() < 2 {
            Zero::zero()
        } else {
            let mean = self.mean();
            let mut v: T = Zero::zero();
            for s in self.iter() {
                let x = *s - mean;
                v = v + x*x;
            }
            // NB: this is _supposed to be_ len-1, not len. If you
            // change it back to len, you will be calculating a
            // population variance, not a sample variance.
            let denom = FromPrimitive::from_uint(self.len()-1).unwrap();
            v/denom
        }
    }

    fn std_dev(self) -> T {
        self.var().sqrt()
    }

    fn std_dev_pct(self) -> T {
        let hundred = FromPrimitive::from_uint(100).unwrap();
        (self.std_dev() / self.mean()) * hundred
    }

    fn median_abs_dev(self) -> T {
        let med = self.median();
        let abs_devs: Vec<T> = self.iter().map(|&v| num::abs(med - v)).collect();
        // This constant is derived by smarter statistics brains than me, but it is
        // consistent with how R and other packages treat the MAD.
        let number = FromPrimitive::from_f64(1.4826).unwrap();
        abs_devs.as_slice().median() * number
    }

    fn median_abs_dev_pct(self) -> T {
        let hundred = FromPrimitive::from_uint(100).unwrap();
        (self.median_abs_dev() / self.median()) * hundred
    }

    fn percentile(self, pct: T) -> T {
        let mut tmp = self.to_vec();
        local_sort(tmp.as_mut_slice());
        percentile_of_sorted(tmp.as_slice(), pct)
    }

    fn quartiles(self) -> (T,T,T) {
        let mut tmp = self.to_vec();
        local_sort(tmp.as_mut_slice());
        let first = FromPrimitive::from_uint(25).unwrap();
        let a = percentile_of_sorted(tmp.as_slice(), first);
        let secound = FromPrimitive::from_uint(50).unwrap();
        let b = percentile_of_sorted(tmp.as_slice(), secound);
        let third = FromPrimitive::from_uint(75).unwrap();
        let c = percentile_of_sorted(tmp.as_slice(), third);
        (a,b,c)
    }

    fn iqr(self) -> T {
        let (a,_,c) = self.quartiles();
        c - a
    }
}


// Helper function: extract a value representing the `pct` percentile of a sorted sample-set, using
// linear interpolation. If samples are not sorted, return nonsensical value.
fn percentile_of_sorted<T: Float + FromPrimitive>(sorted_samples: &[T],
                                                             pct: T) -> T {
    assert!(sorted_samples.len() != 0);
    if sorted_samples.len() == 1 {
        return sorted_samples[0];
    }
    let zero: T = Zero::zero();
    assert!(zero <= pct);
    let hundred = FromPrimitive::from_uint(100).unwrap();
    assert!(pct <= hundred);
    if pct == hundred {
        return sorted_samples[sorted_samples.len() - 1];
    }
    let length = FromPrimitive::from_uint(sorted_samples.len() - 1).unwrap();
    let rank = (pct / hundred) * length;
    let lrank = rank.floor();
    let d = rank - lrank;
    let n = lrank.to_uint().unwrap();
    let lo = sorted_samples[n];
    let hi = sorted_samples[n+1];
    lo + (hi - lo) * d
}


/// Winsorize a set of samples, replacing values above the `100-pct` percentile and below the `pct`
/// percentile with those percentiles themselves. This is a way of minimizing the effect of
/// outliers, at the cost of biasing the sample. It differs from trimming in that it does not
/// change the number of samples, just changes the values of those that are outliers.
///
/// See: http://en.wikipedia.org/wiki/Winsorising
pub fn winsorize<T: Float + FromPrimitive>(samples: &mut [T], pct: T) {
    let mut tmp = samples.to_vec();
    local_sort(tmp.as_mut_slice());
    let lo = percentile_of_sorted(tmp.as_slice(), pct);
    let hundred: T = FromPrimitive::from_uint(100).unwrap();
    let hi = percentile_of_sorted(tmp.as_slice(), hundred-pct);
    for samp in samples.iter_mut() {
        if *samp > hi {
            *samp = hi
        } else if *samp < lo {
            *samp = lo
        }
    }
}

/// Render writes the min, max and quartiles of the provided `Summary` to the provided `Writer`.
pub fn write_5_number_summary<T: Float + Show>(w: &mut io::Writer,
                                               s: &Summary<T>) -> io::IoResult<()> {
    let (q1,q2,q3) = s.quartiles;
    write!(w, "(min={}, q1={}, med={}, q3={}, max={})",
                     s.min,
                     q1,
                     q2,
                     q3,
                     s.max)
}

/// Render a boxplot to the provided writer. The boxplot shows the min, max and quartiles of the
/// provided `Summary` (thus includes the mean) and is scaled to display within the range of the
/// nearest multiple-of-a-power-of-ten above and below the min and max of possible values, and
/// target `width_hint` characters of display (though it will be wider if necessary).
///
/// As an example, the summary with 5-number-summary `(min=15, q1=17, med=20, q3=24, max=31)` might
/// display as:
///
/// ```{.ignore}
///   10 |        [--****#******----------]          | 40
/// ```
pub fn write_boxplot<T: Float + Show + FromPrimitive>(
                     w: &mut io::Writer,
                     s: &Summary<T>,
                     width_hint: uint)
                      -> io::IoResult<()> {

    let (q1,q2,q3) = s.quartiles;

    // the .abs() handles the case where numbers are negative
    let ten: T = FromPrimitive::from_uint(10).unwrap();
    let lomag = ten.powf(s.min.abs().log10().floor());
    let himag = ten.powf(s.max.abs().log10().floor());

    // need to consider when the limit is zero
    let zero: T = Zero::zero();
    let lo = if lomag.is_zero() {
        zero
    } else {
        (s.min / lomag).floor() * lomag
    };

    let hi = if himag.is_zero() {
        zero
    } else {
        (s.max / himag).ceil() * himag
    };

    let range = hi - lo;

    let lostr = lo.to_string();
    let histr = hi.to_string();

    let overhead_width = lostr.len() + histr.len() + 4;
    let range_width = width_hint - overhead_width;
    let range_float = FromPrimitive::from_uint(range_width).unwrap();
    let char_step = range / range_float;

    try!(write!(w, "{} |", lostr));

    let mut c = 0;
    let mut v = lo;

    while c < range_width && v < s.min {
        try!(write!(w, " "));
        v = v + char_step;
        c += 1;
    }
    try!(write!(w, "["));
    c += 1;
    while c < range_width && v < q1 {
        try!(write!(w, "-"));
        v = v + char_step;
        c += 1;
    }
    while c < range_width && v < q2 {
        try!(write!(w, "*"));
        v = v + char_step;
        c += 1;
    }
    try!(write!(w, "#"));
    c += 1;
    while c < range_width && v < q3 {
        try!(write!(w, "*"));
        v = v + char_step;
        c += 1;
    }
    while c < range_width && v < s.max {
        try!(write!(w, "-"));
        v = v + char_step;
        c += 1;
    }
    try!(write!(w, "]"));
    while c < range_width {
        try!(write!(w, " "));
        v = v + char_step;
        c += 1;
    }

    try!(write!(w, "| {}", histr));
    Ok(())
}

/// Returns a HashMap with the number of occurrences of every element in the
/// sequence that the iterator exposes.
pub fn freq_count<T: Iterator<U>, U: Eq+Hash>(mut iter: T) -> hashmap::HashMap<U, uint> {
    let mut map: hashmap::HashMap<U,uint> = hashmap::HashMap::new();
    for elem in iter {
        map.insert_or_update_with(elem, 1, |_, count| *count += 1);
    }
    map
}

// Test vectors generated from R, using the script src/etc/stat-test-vectors.r.

#[cfg(test)]
mod tests {
    use stats::Stats;
    use stats::Summary;
    use stats::write_5_number_summary;
    use stats::write_boxplot;
    use std::io;
    use std::f64;

    macro_rules! assert_approx_eq(
        ($a:expr, $b:expr) => ({
            let (a, b) = (&$a, &$b);
            assert!((*a - *b).abs() < 1.0e-6,
                    "{} is not approximately equal to {}", *a, *b);
        })
    )

    fn check(samples: &[f64], summ: &Summary<f64>) {

        let summ2 = Summary::new(samples);

        let mut w = io::stdout();
        let w = &mut w as &mut io::Writer;
        (write!(w, "\n")).unwrap();
        write_5_number_summary(w, &summ2).unwrap();
        (write!(w, "\n")).unwrap();
        write_boxplot(w, &summ2, 50).unwrap();
        (write!(w, "\n")).unwrap();

        assert_eq!(summ.sum, summ2.sum);
        assert_eq!(summ.min, summ2.min);
        assert_eq!(summ.max, summ2.max);
        assert_eq!(summ.mean, summ2.mean);
        assert_eq!(summ.median, summ2.median);

        // We needed a few more digits to get exact equality on these
        // but they're within float epsilon, which is 1.0e-6.
        assert_approx_eq!(summ.var, summ2.var);
        assert_approx_eq!(summ.std_dev, summ2.std_dev);
        assert_approx_eq!(summ.std_dev_pct, summ2.std_dev_pct);
        assert_approx_eq!(summ.median_abs_dev, summ2.median_abs_dev);
        assert_approx_eq!(summ.median_abs_dev_pct, summ2.median_abs_dev_pct);

        assert_eq!(summ.quartiles, summ2.quartiles);
        assert_eq!(summ.iqr, summ2.iqr);
    }

    #[test]
    fn test_min_max_nan() {
        let xs = &[1.0, 2.0, f64::NAN, 3.0, 4.0];
        let summary = Summary::new(xs);
        assert_eq!(summary.min, 1.0);
        assert_eq!(summary.max, 4.0);
    }

    #[test]
    fn test_norm2() {
        let val = &[
            958.0000000000,
            924.0000000000,
        ];
        let summ = &Summary {
            sum: 1882.0000000000,
            min: 924.0000000000,
            max: 958.0000000000,
            mean: 941.0000000000,
            median: 941.0000000000,
            var: 578.0000000000,
            std_dev: 24.0416305603,
            std_dev_pct: 2.5549022912,
            median_abs_dev: 25.2042000000,
            median_abs_dev_pct: 2.6784484591,
            quartiles: (932.5000000000,941.0000000000,949.5000000000),
            iqr: 17.0000000000,
        };
        check(val, summ);
    }
    #[test]
    fn test_norm10narrow() {
        let val = &[
            966.0000000000,
            985.0000000000,
            1110.0000000000,
            848.0000000000,
            821.0000000000,
            975.0000000000,
            962.0000000000,
            1157.0000000000,
            1217.0000000000,
            955.0000000000,
        ];
        let summ = &Summary {
            sum: 9996.0000000000,
            min: 821.0000000000,
            max: 1217.0000000000,
            mean: 999.6000000000,
            median: 970.5000000000,
            var: 16050.7111111111,
            std_dev: 126.6914010938,
            std_dev_pct: 12.6742097933,
            median_abs_dev: 102.2994000000,
            median_abs_dev_pct: 10.5408964451,
            quartiles: (956.7500000000,970.5000000000,1078.7500000000),
            iqr: 122.0000000000,
        };
        check(val, summ);
    }
    #[test]
    fn test_norm10medium() {
        let val = &[
            954.0000000000,
            1064.0000000000,
            855.0000000000,
            1000.0000000000,
            743.0000000000,
            1084.0000000000,
            704.0000000000,
            1023.0000000000,
            357.0000000000,
            869.0000000000,
        ];
        let summ = &Summary {
            sum: 8653.0000000000,
            min: 357.0000000000,
            max: 1084.0000000000,
            mean: 865.3000000000,
            median: 911.5000000000,
            var: 48628.4555555556,
            std_dev: 220.5186059170,
            std_dev_pct: 25.4846418487,
            median_abs_dev: 195.7032000000,
            median_abs_dev_pct: 21.4704552935,
            quartiles: (771.0000000000,911.5000000000,1017.2500000000),
            iqr: 246.2500000000,
        };
        check(val, summ);
    }
    #[test]
    fn test_norm10wide() {
        let val = &[
            505.0000000000,
            497.0000000000,
            1591.0000000000,
            887.0000000000,
            1026.0000000000,
            136.0000000000,
            1580.0000000000,
            940.0000000000,
            754.0000000000,
            1433.0000000000,
        ];
        let summ = &Summary {
            sum: 9349.0000000000,
            min: 136.0000000000,
            max: 1591.0000000000,
            mean: 934.9000000000,
            median: 913.5000000000,
            var: 239208.9888888889,
            std_dev: 489.0899599142,
            std_dev_pct: 52.3146817750,
            median_abs_dev: 611.5725000000,
            median_abs_dev_pct: 66.9482758621,
            quartiles: (567.2500000000,913.5000000000,1331.2500000000),
            iqr: 764.0000000000,
        };
        check(val, summ);
    }
    #[test]
    fn test_norm25verynarrow() {
        let val = &[
            991.0000000000,
            1018.0000000000,
            998.0000000000,
            1013.0000000000,
            974.0000000000,
            1007.0000000000,
            1014.0000000000,
            999.0000000000,
            1011.0000000000,
            978.0000000000,
            985.0000000000,
            999.0000000000,
            983.0000000000,
            982.0000000000,
            1015.0000000000,
            1002.0000000000,
            977.0000000000,
            948.0000000000,
            1040.0000000000,
            974.0000000000,
            996.0000000000,
            989.0000000000,
            1015.0000000000,
            994.0000000000,
            1024.0000000000,
        ];
        let summ = &Summary {
            sum: 24926.0000000000,
            min: 948.0000000000,
            max: 1040.0000000000,
            mean: 997.0400000000,
            median: 998.0000000000,
            var: 393.2066666667,
            std_dev: 19.8294393937,
            std_dev_pct: 1.9888308788,
            median_abs_dev: 22.2390000000,
            median_abs_dev_pct: 2.2283567134,
            quartiles: (983.0000000000,998.0000000000,1013.0000000000),
            iqr: 30.0000000000,
        };
        check(val, summ);
    }
    #[test]
    fn test_exp10a() {
        let val = &[
            23.0000000000,
            11.0000000000,
            2.0000000000,
            57.0000000000,
            4.0000000000,
            12.0000000000,
            5.0000000000,
            29.0000000000,
            3.0000000000,
            21.0000000000,
        ];
        let summ = &Summary {
            sum: 167.0000000000,
            min: 2.0000000000,
            max: 57.0000000000,
            mean: 16.7000000000,
            median: 11.5000000000,
            var: 287.7888888889,
            std_dev: 16.9643416875,
            std_dev_pct: 101.5828843560,
            median_abs_dev: 13.3434000000,
            median_abs_dev_pct: 116.0295652174,
            quartiles: (4.2500000000,11.5000000000,22.5000000000),
            iqr: 18.2500000000,
        };
        check(val, summ);
    }
    #[test]
    fn test_exp10b() {
        let val = &[
            24.0000000000,
            17.0000000000,
            6.0000000000,
            38.0000000000,
            25.0000000000,
            7.0000000000,
            51.0000000000,
            2.0000000000,
            61.0000000000,
            32.0000000000,
        ];
        let summ = &Summary {
            sum: 263.0000000000,
            min: 2.0000000000,
            max: 61.0000000000,
            mean: 26.3000000000,
            median: 24.5000000000,
            var: 383.5666666667,
            std_dev: 19.5848580967,
            std_dev_pct: 74.4671410520,
            median_abs_dev: 22.9803000000,
            median_abs_dev_pct: 93.7971428571,
            quartiles: (9.5000000000,24.5000000000,36.5000000000),
            iqr: 27.0000000000,
        };
        check(val, summ);
    }
    #[test]
    fn test_exp10c() {
        let val = &[
            71.0000000000,
            2.0000000000,
            32.0000000000,
            1.0000000000,
            6.0000000000,
            28.0000000000,
            13.0000000000,
            37.0000000000,
            16.0000000000,
            36.0000000000,
        ];
        let summ = &Summary {
            sum: 242.0000000000,
            min: 1.0000000000,
            max: 71.0000000000,
            mean: 24.2000000000,
            median: 22.0000000000,
            var: 458.1777777778,
            std_dev: 21.4050876611,
            std_dev_pct: 88.4507754589,
            median_abs_dev: 21.4977000000,
            median_abs_dev_pct: 97.7168181818,
            quartiles: (7.7500000000,22.0000000000,35.0000000000),
            iqr: 27.2500000000,
        };
        check(val, summ);
    }
    #[test]
    fn test_exp25() {
        let val = &[
            3.0000000000,
            24.0000000000,
            1.0000000000,
            19.0000000000,
            7.0000000000,
            5.0000000000,
            30.0000000000,
            39.0000000000,
            31.0000000000,
            13.0000000000,
            25.0000000000,
            48.0000000000,
            1.0000000000,
            6.0000000000,
            42.0000000000,
            63.0000000000,
            2.0000000000,
            12.0000000000,
            108.0000000000,
            26.0000000000,
            1.0000000000,
            7.0000000000,
            44.0000000000,
            25.0000000000,
            11.0000000000,
        ];
        let summ = &Summary {
            sum: 593.0000000000,
            min: 1.0000000000,
            max: 108.0000000000,
            mean: 23.7200000000,
            median: 19.0000000000,
            var: 601.0433333333,
            std_dev: 24.5161851301,
            std_dev_pct: 103.3565983562,
            median_abs_dev: 19.2738000000,
            median_abs_dev_pct: 101.4410526316,
            quartiles: (6.0000000000,19.0000000000,31.0000000000),
            iqr: 25.0000000000,
        };
        check(val, summ);
    }
    #[test]
    fn test_binom25() {
        let val = &[
            18.0000000000,
            17.0000000000,
            27.0000000000,
            15.0000000000,
            21.0000000000,
            25.0000000000,
            17.0000000000,
            24.0000000000,
            25.0000000000,
            24.0000000000,
            26.0000000000,
            26.0000000000,
            23.0000000000,
            15.0000000000,
            23.0000000000,
            17.0000000000,
            18.0000000000,
            18.0000000000,
            21.0000000000,
            16.0000000000,
            15.0000000000,
            31.0000000000,
            20.0000000000,
            17.0000000000,
            15.0000000000,
        ];
        let summ = &Summary {
            sum: 514.0000000000,
            min: 15.0000000000,
            max: 31.0000000000,
            mean: 20.5600000000,
            median: 20.0000000000,
            var: 20.8400000000,
            std_dev: 4.5650848842,
            std_dev_pct: 22.2037202539,
            median_abs_dev: 5.9304000000,
            median_abs_dev_pct: 29.6520000000,
            quartiles: (17.0000000000,20.0000000000,24.0000000000),
            iqr: 7.0000000000,
        };
        check(val, summ);
    }
    #[test]
    fn test_pois25lambda30() {
        let val = &[
            27.0000000000,
            33.0000000000,
            34.0000000000,
            34.0000000000,
            24.0000000000,
            39.0000000000,
            28.0000000000,
            27.0000000000,
            31.0000000000,
            28.0000000000,
            38.0000000000,
            21.0000000000,
            33.0000000000,
            36.0000000000,
            29.0000000000,
            37.0000000000,
            32.0000000000,
            34.0000000000,
            31.0000000000,
            39.0000000000,
            25.0000000000,
            31.0000000000,
            32.0000000000,
            40.0000000000,
            24.0000000000,
        ];
        let summ = &Summary {
            sum: 787.0000000000,
            min: 21.0000000000,
            max: 40.0000000000,
            mean: 31.4800000000,
            median: 32.0000000000,
            var: 26.5933333333,
            std_dev: 5.1568724372,
            std_dev_pct: 16.3814245145,
            median_abs_dev: 5.9304000000,
            median_abs_dev_pct: 18.5325000000,
            quartiles: (28.0000000000,32.0000000000,34.0000000000),
            iqr: 6.0000000000,
        };
        check(val, summ);
    }
    #[test]
    fn test_pois25lambda40() {
        let val = &[
            42.0000000000,
            50.0000000000,
            42.0000000000,
            46.0000000000,
            34.0000000000,
            45.0000000000,
            34.0000000000,
            49.0000000000,
            39.0000000000,
            28.0000000000,
            40.0000000000,
            35.0000000000,
            37.0000000000,
            39.0000000000,
            46.0000000000,
            44.0000000000,
            32.0000000000,
            45.0000000000,
            42.0000000000,
            37.0000000000,
            48.0000000000,
            42.0000000000,
            33.0000000000,
            42.0000000000,
            48.0000000000,
        ];
        let summ = &Summary {
            sum: 1019.0000000000,
            min: 28.0000000000,
            max: 50.0000000000,
            mean: 40.7600000000,
            median: 42.0000000000,
            var: 34.4400000000,
            std_dev: 5.8685603004,
            std_dev_pct: 14.3978417577,
            median_abs_dev: 5.9304000000,
            median_abs_dev_pct: 14.1200000000,
            quartiles: (37.0000000000,42.0000000000,45.0000000000),
            iqr: 8.0000000000,
        };
        check(val, summ);
    }
    #[test]
    fn test_pois25lambda50() {
        let val = &[
            45.0000000000,
            43.0000000000,
            44.0000000000,
            61.0000000000,
            51.0000000000,
            53.0000000000,
            59.0000000000,
            52.0000000000,
            49.0000000000,
            51.0000000000,
            51.0000000000,
            50.0000000000,
            49.0000000000,
            56.0000000000,
            42.0000000000,
            52.0000000000,
            51.0000000000,
            43.0000000000,
            48.0000000000,
            48.0000000000,
            50.0000000000,
            42.0000000000,
            43.0000000000,
            42.0000000000,
            60.0000000000,
        ];
        let summ = &Summary {
            sum: 1235.0000000000,
            min: 42.0000000000,
            max: 61.0000000000,
            mean: 49.4000000000,
            median: 50.0000000000,
            var: 31.6666666667,
            std_dev: 5.6273143387,
            std_dev_pct: 11.3913245723,
            median_abs_dev: 4.4478000000,
            median_abs_dev_pct: 8.8956000000,
            quartiles: (44.0000000000,50.0000000000,52.0000000000),
            iqr: 8.0000000000,
        };
        check(val, summ);
    }
    #[test]
    fn test_unif25() {
        let val = &[
            99.0000000000,
            55.0000000000,
            92.0000000000,
            79.0000000000,
            14.0000000000,
            2.0000000000,
            33.0000000000,
            49.0000000000,
            3.0000000000,
            32.0000000000,
            84.0000000000,
            59.0000000000,
            22.0000000000,
            86.0000000000,
            76.0000000000,
            31.0000000000,
            29.0000000000,
            11.0000000000,
            41.0000000000,
            53.0000000000,
            45.0000000000,
            44.0000000000,
            98.0000000000,
            98.0000000000,
            7.0000000000,
        ];
        let summ = &Summary {
            sum: 1242.0000000000,
            min: 2.0000000000,
            max: 99.0000000000,
            mean: 49.6800000000,
            median: 45.0000000000,
            var: 1015.6433333333,
            std_dev: 31.8691595957,
            std_dev_pct: 64.1488719719,
            median_abs_dev: 45.9606000000,
            median_abs_dev_pct: 102.1346666667,
            quartiles: (29.0000000000,45.0000000000,79.0000000000),
            iqr: 50.0000000000,
        };
        check(val, summ);
    }

    #[test]
    fn test_boxplot_nonpositive() {
        fn t(s: &Summary<f64>, expected: String) {
            use std::io::MemWriter;
            let mut m = MemWriter::new();
            write_boxplot(&mut m as &mut io::Writer, s, 30).unwrap();
            let out = String::from_utf8(m.unwrap()).unwrap();
            assert_eq!(out, expected);
        }

        t(&Summary::new([-2.0f64, -1.0f64]),
                        "-2 |[------******#*****---]| -1".to_string());
        t(&Summary::new([0.0f64, 2.0f64]),
                        "0 |[-------*****#*******---]| 2".to_string());
        t(&Summary::new([-2.0f64, 0.0f64]),
                        "-2 |[------******#******---]| 0".to_string());

    }
    #[test]
    fn test_sum_f64s() {
        assert_eq!([0.5f64, 3.2321f64, 1.5678f64].sum(), 5.2999);
    }
    #[test]
    fn test_sum_f64_between_ints_that_sum_to_0() {
        assert_eq!([1e30f64, 1.2f64, -1e30f64].sum(), 1.2);
    }
}

#[cfg(test)]
mod bench {
    use Bencher;
    use stats::Stats;

    #[bench]
    pub fn sum_three_items(b: &mut Bencher) {
        b.iter(|| {
            [1e20f64, 1.5f64, -1e20f64].sum();
        })
    }
    #[bench]
    pub fn sum_many_f64(b: &mut Bencher) {
        let nums = [-1e30f64, 1e60, 1e30, 1.0, -1e60];
        let v = Vec::from_fn(500, |i| nums[i%5]);

        b.iter(|| {
            v.as_slice().sum();
        })
    }
}
