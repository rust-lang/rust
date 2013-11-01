// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
Sampling from random distributions.

This is a generalization of `Rand` to allow parameters to control the
exact properties of the generated values, e.g. the mean and standard
deviation of a normal distribution. The `Sample` trait is the most
general, and allows for generating values that change some state
internally. The `IndependentSample` trait is for generating values
that do not need to record state.

*/

use iter::range;
use option::{Some, None};
use num;
use rand::{Rng,Rand};
use clone::Clone;

pub use self::range::Range;
pub use self::gamma::Gamma;

pub mod range;
pub mod gamma;

/// Types that can be used to create a random instance of `Support`.
pub trait Sample<Support> {
    /// Generate a random value of `Support`, using `rng` as the
    /// source of randomness.
    fn sample<R: Rng>(&mut self, rng: &mut R) -> Support;
}

/// `Sample`s that do not require keeping track of state.
///
/// Since no state is recored, each sample is (statistically)
/// independent of all others, assuming the `Rng` used has this
/// property.
// XXX maybe having this separate is overkill (the only reason is to
// take &self rather than &mut self)? or maybe this should be the
// trait called `Sample` and the other should be `DependentSample`.
pub trait IndependentSample<Support>: Sample<Support> {
    /// Generate a random value.
    fn ind_sample<R: Rng>(&self, &mut R) -> Support;
}

/// A wrapper for generating types that implement `Rand` via the
/// `Sample` & `IndependentSample` traits.
pub struct RandSample<Sup>;

impl<Sup: Rand> Sample<Sup> for RandSample<Sup> {
    fn sample<R: Rng>(&mut self, rng: &mut R) -> Sup { self.ind_sample(rng) }
}

impl<Sup: Rand> IndependentSample<Sup> for RandSample<Sup> {
    fn ind_sample<R: Rng>(&self, rng: &mut R) -> Sup {
        rng.gen()
    }
}

/// A value with a particular weight for use with `WeightedChoice`.
pub struct Weighted<T> {
    /// The numerical weight of this item
    weight: uint,
    /// The actual item which is being weighted
    item: T,
}

/// A distribution that selects from a finite collection of weighted items.
///
/// Each item has an associated weight that influences how likely it
/// is to be chosen: higher weight is more likely.
///
/// The `Clone` restriction is a limitation of the `Sample` and
/// `IndepedentSample` traits. Note that `&T` is (cheaply) `Clone` for
/// all `T`, as is `uint`, so one can store references or indices into
/// another vector.
///
/// # Example
///
/// ```rust
/// use std::rand;
/// use std::rand::distributions::{Weighted, WeightedChoice, IndepedentSample};
///
/// fn main() {
///     let wc = WeightedChoice::new(~[Weighted { weight: 2, item: 'a' },
///                                    Weighted { weight: 4, item: 'b' },
///                                    Weighted { weight: 1, item: 'c' }]);
///     let rng = rand::task_rng();
///     for _ in range(0, 16) {
///          // on average prints 'a' 4 times, 'b' 8 and 'c' twice.
///          println!("{}", wc.ind_sample(rng));
///     }
/// }
/// ```
pub struct WeightedChoice<T> {
    priv items: ~[Weighted<T>],
    priv weight_range: Range<uint>
}

impl<T: Clone> WeightedChoice<T> {
    /// Create a new `WeightedChoice`.
    ///
    /// Fails if:
    /// - `v` is empty
    /// - the total weight is 0
    /// - the total weight is larger than a `uint` can contain.
    pub fn new(mut items: ~[Weighted<T>]) -> WeightedChoice<T> {
        // strictly speaking, this is subsumed by the total weight == 0 case
        assert!(!items.is_empty(), "WeightedChoice::new called with no items");

        let mut running_total = 0u;

        // we convert the list from individual weights to cumulative
        // weights so we can binary search. This *could* drop elements
        // with weight == 0 as an optimisation.
        for item in items.mut_iter() {
            running_total = running_total.checked_add(&item.weight)
                .expect("WeightedChoice::new called with a total weight larger \
                        than a uint can contain");

            item.weight = running_total;
        }
        assert!(running_total != 0, "WeightedChoice::new called with a total weight of 0");

        WeightedChoice {
            items: items,
            // we're likely to be generating numbers in this range
            // relatively often, so might as well cache it
            weight_range: Range::new(0, running_total)
        }
    }
}

impl<T: Clone> Sample<T> for WeightedChoice<T> {
    fn sample<R: Rng>(&mut self, rng: &mut R) -> T { self.ind_sample(rng) }
}

impl<T: Clone> IndependentSample<T> for WeightedChoice<T> {
    fn ind_sample<R: Rng>(&self, rng: &mut R) -> T {
        // we want to find the first element that has cumulative
        // weight > sample_weight, which we do by binary since the
        // cumulative weights of self.items are sorted.

        // choose a weight in [0, total_weight)
        let sample_weight = self.weight_range.ind_sample(rng);

        // short circuit when it's the first item
        if sample_weight < self.items[0].weight {
            return self.items[0].item.clone();
        }

        let mut idx = 0;
        let mut modifier = self.items.len();

        // now we know that every possibility has an element to the
        // left, so we can just search for the last element that has
        // cumulative weight <= sample_weight, then the next one will
        // be "it". (Note that this greatest element will never be the
        // last element of the vector, since sample_weight is chosen
        // in [0, total_weight) and the cumulative weight of the last
        // one is exactly the total weight.)
        while modifier > 1 {
            let i = idx + modifier / 2;
            if self.items[i].weight <= sample_weight {
                // we're small, so look to the right, but allow this
                // exact element still.
                idx = i;
                // we need the `/ 2` to round up otherwise we'll drop
                // the trailing elements when `modifier` is odd.
                modifier += 1;
            } else {
                // otherwise we're too big, so go left. (i.e. do
                // nothing)
            }
            modifier /= 2;
        }
        return self.items[idx + 1].item.clone();
    }
}

mod ziggurat_tables;

/// Sample a random number using the Ziggurat method (specifically the
/// ZIGNOR variant from Doornik 2005). Most of the arguments are
/// directly from the paper:
///
/// * `rng`: source of randomness
/// * `symmetric`: whether this is a symmetric distribution, or one-sided with P(x < 0) = 0.
/// * `X`: the $x_i$ abscissae.
/// * `F`: precomputed values of the PDF at the $x_i$, (i.e. $f(x_i)$)
/// * `F_DIFF`: precomputed values of $f(x_i) - f(x_{i+1})$
/// * `pdf`: the probability density function
/// * `zero_case`: manual sampling from the tail when we chose the
///    bottom box (i.e. i == 0)

// the perf improvement (25-50%) is definitely worth the extra code
// size from force-inlining.
#[inline(always)]
fn ziggurat<R:Rng>(rng: &mut R,
                   symmetric: bool,
                   X: ziggurat_tables::ZigTable,
                   F: ziggurat_tables::ZigTable,
                   pdf: &'static fn(f64) -> f64,
                   zero_case: &'static fn(&mut R, f64) -> f64) -> f64 {
    static SCALE: f64 = (1u64 << 53) as f64;
    loop {
        // reimplement the f64 generation as an optimisation suggested
        // by the Doornik paper: we have a lot of precision-space
        // (i.e. there are 11 bits of the 64 of a u64 to use after
        // creating a f64), so we might as well reuse some to save
        // generating a whole extra random number. (Seems to be 15%
        // faster.)
        let bits: u64 = rng.gen();
        let i = (bits & 0xff) as uint;
        let f = (bits >> 11) as f64 / SCALE;

        // u is either U(-1, 1) or U(0, 1) depending on if this is a
        // symmetric distribution or not.
        let u = if symmetric {2.0 * f - 1.0} else {f};
        let x = u * X[i];

        let test_x = if symmetric {num::abs(x)} else {x};

        // algebraically equivalent to |u| < X[i+1]/X[i] (or u < X[i+1]/X[i])
        if test_x < X[i + 1] {
            return x;
        }
        if i == 0 {
            return zero_case(rng, u);
        }
        // algebraically equivalent to f1 + DRanU()*(f0 - f1) < 1
        if F[i + 1] + (F[i] - F[i + 1]) * rng.gen() < pdf(x) {
            return x;
        }
    }
}

/// A wrapper around an `f64` to generate N(0, 1) random numbers
/// (a.k.a.  a standard normal, or Gaussian).
///
/// See `Normal` for the general normal distribution. That this has to
/// be unwrapped before use as an `f64` (using either `*` or
/// `cast::transmute` is safe).
///
/// Implemented via the ZIGNOR variant[1] of the Ziggurat method.
///
/// [1]: Jurgen A. Doornik (2005). [*An Improved Ziggurat Method to
/// Generate Normal Random
/// Samples*](http://www.doornik.com/research/ziggurat.pdf). Nuffield
/// College, Oxford
pub struct StandardNormal(f64);

impl Rand for StandardNormal {
    fn rand<R:Rng>(rng: &mut R) -> StandardNormal {
        #[inline]
        fn pdf(x: f64) -> f64 {
            ((-x*x/2.0) as f64).exp()
        }
        #[inline]
        fn zero_case<R:Rng>(rng: &mut R, u: f64) -> f64 {
            // compute a random number in the tail by hand

            // strange initial conditions, because the loop is not
            // do-while, so the condition should be true on the first
            // run, they get overwritten anyway (0 < 1, so these are
            // good).
            let mut x = 1.0f64;
            let mut y = 0.0f64;

            // FIXME #7755: infinities?
            while -2.0 * y < x * x {
                x = rng.gen::<f64>().ln() / ziggurat_tables::ZIG_NORM_R;
                y = rng.gen::<f64>().ln();
            }

            if u < 0.0 { x - ziggurat_tables::ZIG_NORM_R } else { ziggurat_tables::ZIG_NORM_R - x }
        }

        StandardNormal(ziggurat(
            rng,
            true, // this is symmetric
            &ziggurat_tables::ZIG_NORM_X,
            &ziggurat_tables::ZIG_NORM_F,
            pdf, zero_case))
    }
}

/// The normal distribution `N(mean, std_dev**2)`.
///
/// This uses the ZIGNOR variant of the Ziggurat method, see
/// `StandardNormal` for more details.
///
/// # Example
///
/// ```
/// use std::rand;
/// use std::rand::distributions::{Normal, IndependentSample};
///
/// fn main() {
///     let normal = Normal::new(2.0, 3.0);
///     let v = normal.ind_sample(rand::task_rng());
///     println!("{} is from a N(2, 9) distribution", v)
/// }
/// ```
pub struct Normal {
    priv mean: f64,
    priv std_dev: f64
}

impl Normal {
    /// Construct a new `Normal` distribution with the given mean and
    /// standard deviation. Fails if `std_dev < 0`.
    pub fn new(mean: f64, std_dev: f64) -> Normal {
        assert!(std_dev >= 0.0, "Normal::new called with `std_dev` < 0");
        Normal {
            mean: mean,
            std_dev: std_dev
        }
    }
}
impl Sample<f64> for Normal {
    fn sample<R: Rng>(&mut self, rng: &mut R) -> f64 { self.ind_sample(rng) }
}
impl IndependentSample<f64> for Normal {
    fn ind_sample<R: Rng>(&self, rng: &mut R) -> f64 {
        self.mean + self.std_dev * (*rng.gen::<StandardNormal>())
    }
}

/// A wrapper around an `f64` to generate Exp(1) random numbers.
///
/// See `Exp` for the general exponential distribution.Note that this
 // has to be unwrapped before use as an `f64` (using either
/// `*` or `cast::transmute` is safe).
///
/// Implemented via the ZIGNOR variant[1] of the Ziggurat method. The
/// exact description in the paper was adjusted to use tables for the
/// exponential distribution rather than normal.
///
/// [1]: Jurgen A. Doornik (2005). [*An Improved Ziggurat Method to
/// Generate Normal Random
/// Samples*](http://www.doornik.com/research/ziggurat.pdf). Nuffield
/// College, Oxford
pub struct Exp1(f64);

// This could be done via `-rng.gen::<f64>().ln()` but that is slower.
impl Rand for Exp1 {
    #[inline]
    fn rand<R:Rng>(rng: &mut R) -> Exp1 {
        #[inline]
        fn pdf(x: f64) -> f64 {
            (-x).exp()
        }
        #[inline]
        fn zero_case<R:Rng>(rng: &mut R, _u: f64) -> f64 {
            ziggurat_tables::ZIG_EXP_R - rng.gen::<f64>().ln()
        }

        Exp1(ziggurat(rng, false,
                      &ziggurat_tables::ZIG_EXP_X,
                      &ziggurat_tables::ZIG_EXP_F,
                      pdf, zero_case))
    }
}

/// The exponential distribution `Exp(lambda)`.
///
/// This distribution has density function: `f(x) = lambda *
/// exp(-lambda * x)` for `x > 0`.
///
/// # Example
///
/// ```
/// use std::rand;
/// use std::rand::distributions::{Exp, IndependentSample};
///
/// fn main() {
///     let exp = Exp::new(2.0);
///     let v = exp.ind_sample(rand::task_rng());
///     println!("{} is from a Exp(2) distribution", v);
/// }
/// ```
pub struct Exp {
    /// `lambda` stored as `1/lambda`, since this is what we scale by.
    priv lambda_inverse: f64
}

impl Exp {
    /// Construct a new `Exp` with the given shape parameter
    /// `lambda`. Fails if `lambda <= 0`.
    pub fn new(lambda: f64) -> Exp {
        assert!(lambda > 0.0, "Exp::new called with `lambda` <= 0");
        Exp { lambda_inverse: 1.0 / lambda }
    }
}

impl Sample<f64> for Exp {
    fn sample<R: Rng>(&mut self, rng: &mut R) -> f64 { self.ind_sample(rng) }
}
impl IndependentSample<f64> for Exp {
    fn ind_sample<R: Rng>(&self, rng: &mut R) -> f64 {
        (*rng.gen::<Exp1>()) * self.lambda_inverse
    }
}

#[cfg(test)]
mod tests {
    use rand::*;
    use super::*;
    use iter::range;
    use option::{Some, None};

    struct ConstRand(uint);
    impl Rand for ConstRand {
        fn rand<R: Rng>(_: &mut R) -> ConstRand {
            ConstRand(0)
        }
    }

    // 0, 1, 2, 3, ...
    struct CountingRng { i: u32 }
    impl Rng for CountingRng {
        fn next_u32(&mut self) -> u32 {
            self.i += 1;
            self.i - 1
        }
        fn next_u64(&mut self) -> u64 {
            self.next_u32() as u64
        }
    }

    #[test]
    fn test_rand_sample() {
        let mut rand_sample = RandSample::<ConstRand>;

        assert_eq!(*rand_sample.sample(task_rng()), 0);
        assert_eq!(*rand_sample.ind_sample(task_rng()), 0);
    }

    #[test]
    fn test_normal() {
        let mut norm = Normal::new(10.0, 10.0);
        let rng = task_rng();
        for _ in range(0, 1000) {
            norm.sample(rng);
            norm.ind_sample(rng);
        }
    }
    #[test]
    #[should_fail]
    fn test_normal_invalid_sd() {
        Normal::new(10.0, -1.0);
    }

    #[test]
    fn test_exp() {
        let mut exp = Exp::new(10.0);
        let rng = task_rng();
        for _ in range(0, 1000) {
            assert!(exp.sample(rng) >= 0.0);
            assert!(exp.ind_sample(rng) >= 0.0);
        }
    }
    #[test]
    #[should_fail]
    fn test_exp_invalid_lambda_zero() {
        Exp::new(0.0);
    }
    #[test]
    #[should_fail]
    fn test_exp_invalid_lambda_neg() {
        Exp::new(-10.0);
    }

    #[test]
    fn test_weighted_choice() {
        // this makes assumptions about the internal implementation of
        // WeightedChoice, specifically: it doesn't reorder the items,
        // it doesn't do weird things to the RNG (so 0 maps to 0, 1 to
        // 1, internally; modulo a modulo operation).

        macro_rules! t (
            ($items:expr, $expected:expr) => {{
                let wc = WeightedChoice::new($items);
                let expected = $expected;

                let mut rng = CountingRng { i: 0 };

                for &val in expected.iter() {
                    assert_eq!(wc.ind_sample(&mut rng), val)
                }
            }}
        );

        t!(~[Weighted { weight: 1, item: 10}], ~[10]);

        // skip some
        t!(~[Weighted { weight: 0, item: 20},
             Weighted { weight: 2, item: 21},
             Weighted { weight: 0, item: 22},
             Weighted { weight: 1, item: 23}],
           ~[21,21, 23]);

        // different weights
        t!(~[Weighted { weight: 4, item: 30},
             Weighted { weight: 3, item: 31}],
           ~[30,30,30,30, 31,31,31]);

        // check that we're binary searching
        // correctly with some vectors of odd
        // length.
        t!(~[Weighted { weight: 1, item: 40},
             Weighted { weight: 1, item: 41},
             Weighted { weight: 1, item: 42},
             Weighted { weight: 1, item: 43},
             Weighted { weight: 1, item: 44}],
           ~[40, 41, 42, 43, 44]);
        t!(~[Weighted { weight: 1, item: 50},
             Weighted { weight: 1, item: 51},
             Weighted { weight: 1, item: 52},
             Weighted { weight: 1, item: 53},
             Weighted { weight: 1, item: 54},
             Weighted { weight: 1, item: 55},
             Weighted { weight: 1, item: 56}],
           ~[50, 51, 52, 53, 54, 55, 56]);
    }

    #[test] #[should_fail]
    fn test_weighted_choice_no_items() {
        WeightedChoice::<int>::new(~[]);
    }
    #[test] #[should_fail]
    fn test_weighted_choice_zero_weight() {
        WeightedChoice::new(~[Weighted { weight: 0, item: 0},
                              Weighted { weight: 0, item: 1}]);
    }
    #[test] #[should_fail]
    fn test_weighted_choice_weight_overflows() {
        let x = (-1) as uint / 2; // x + x + 2 is the overflow
        WeightedChoice::new(~[Weighted { weight: x, item: 0 },
                              Weighted { weight: 1, item: 1 },
                              Weighted { weight: x, item: 2 },
                              Weighted { weight: 1, item: 3 }]);
    }
}

#[cfg(test)]
mod bench {
    use extra::test::BenchHarness;
    use rand::{XorShiftRng, RAND_BENCH_N};
    use super::*;
    use iter::range;
    use option::{Some, None};
    use mem::size_of;

    #[bench]
    fn rand_normal(bh: &mut BenchHarness) {
        let mut rng = XorShiftRng::new();
        let mut normal = Normal::new(-2.71828, 3.14159);

        do bh.iter {
            for _ in range(0, RAND_BENCH_N) {
                normal.sample(&mut rng);
            }
        }
        bh.bytes = size_of::<f64>() as u64 * RAND_BENCH_N;
    }
    #[bench]
    fn rand_exp(bh: &mut BenchHarness) {
        let mut rng = XorShiftRng::new();
        let mut exp = Exp::new(2.71828 * 3.14159);

        do bh.iter {
            for _ in range(0, RAND_BENCH_N) {
                exp.sample(&mut rng);
            }
        }
        bh.bytes = size_of::<f64>() as u64 * RAND_BENCH_N;
    }
}
