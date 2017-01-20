// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Sampling from random distributions.
//!
//! This is a generalization of `Rand` to allow parameters to control the
//! exact properties of the generated values, e.g. the mean and standard
//! deviation of a normal distribution. The `Sample` trait is the most
//! general, and allows for generating values that change some state
//! internally. The `IndependentSample` trait is for generating values
//! that do not need to record state.

use core::fmt;

#[cfg(not(test))] // only necessary for no_std
use core::num::Float;

use core::marker::PhantomData;

use {Rand, Rng};

pub use self::range::Range;
pub use self::gamma::{ChiSquared, FisherF, Gamma, StudentT};
pub use self::normal::{LogNormal, Normal};
pub use self::exponential::Exp;

pub mod range;
pub mod gamma;
pub mod normal;
pub mod exponential;

/// Types that can be used to create a random instance of `Support`.
pub trait Sample<Support> {
    /// Generate a random value of `Support`, using `rng` as the
    /// source of randomness.
    fn sample<R: Rng>(&mut self, rng: &mut R) -> Support;
}

/// `Sample`s that do not require keeping track of state.
///
/// Since no state is recorded, each sample is (statistically)
/// independent of all others, assuming the `Rng` used has this
/// property.
// FIXME maybe having this separate is overkill (the only reason is to
// take &self rather than &mut self)? or maybe this should be the
// trait called `Sample` and the other should be `DependentSample`.
pub trait IndependentSample<Support>: Sample<Support> {
    /// Generate a random value.
    fn ind_sample<R: Rng>(&self, &mut R) -> Support;
}

/// A wrapper for generating types that implement `Rand` via the
/// `Sample` & `IndependentSample` traits.
pub struct RandSample<Sup> {
    _marker: PhantomData<Sup>,
}

impl<Sup> RandSample<Sup> {
    pub fn new() -> RandSample<Sup> {
        RandSample { _marker: PhantomData }
    }
}

impl<Sup: Rand> Sample<Sup> for RandSample<Sup> {
    fn sample<R: Rng>(&mut self, rng: &mut R) -> Sup {
        self.ind_sample(rng)
    }
}

impl<Sup: Rand> IndependentSample<Sup> for RandSample<Sup> {
    fn ind_sample<R: Rng>(&self, rng: &mut R) -> Sup {
        rng.gen()
    }
}

impl<Sup> fmt::Debug for RandSample<Sup> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("RandSample { .. }")
    }
}

/// A value with a particular weight for use with `WeightedChoice`.
pub struct Weighted<T> {
    /// The numerical weight of this item
    pub weight: usize,
    /// The actual item which is being weighted
    pub item: T,
}

impl<T: fmt::Debug> fmt::Debug for Weighted<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Weighted")
         .field("weight", &self.weight)
         .field("item", &self.item)
         .finish()
    }
}

/// A distribution that selects from a finite collection of weighted items.
///
/// Each item has an associated weight that influences how likely it
/// is to be chosen: higher weight is more likely.
///
/// The `Clone` restriction is a limitation of the `Sample` and
/// `IndependentSample` traits. Note that `&T` is (cheaply) `Clone` for
/// all `T`, as is `usize`, so one can store references or indices into
/// another vector.
pub struct WeightedChoice<'a, T: 'a> {
    items: &'a mut [Weighted<T>],
    weight_range: Range<usize>,
}

impl<'a, T: Clone> WeightedChoice<'a, T> {
    /// Create a new `WeightedChoice`.
    ///
    /// Panics if:
    /// - `v` is empty
    /// - the total weight is 0
    /// - the total weight is larger than a `usize` can contain.
    pub fn new(items: &'a mut [Weighted<T>]) -> WeightedChoice<'a, T> {
        // strictly speaking, this is subsumed by the total weight == 0 case
        assert!(!items.is_empty(),
                "WeightedChoice::new called with no items");

        let mut running_total = 0_usize;

        // we convert the list from individual weights to cumulative
        // weights so we can binary search. This *could* drop elements
        // with weight == 0 as an optimisation.
        for item in &mut *items {
            running_total = match running_total.checked_add(item.weight) {
                Some(n) => n,
                None => {
                    panic!("WeightedChoice::new called with a total weight larger than a usize \
                            can contain")
                }
            };

            item.weight = running_total;
        }
        assert!(running_total != 0,
                "WeightedChoice::new called with a total weight of 0");

        WeightedChoice {
            items: items,
            // we're likely to be generating numbers in this range
            // relatively often, so might as well cache it
            weight_range: Range::new(0, running_total),
        }
    }
}

impl<'a, T: Clone> Sample<T> for WeightedChoice<'a, T> {
    fn sample<R: Rng>(&mut self, rng: &mut R) -> T {
        self.ind_sample(rng)
    }
}

impl<'a, T: Clone> IndependentSample<T> for WeightedChoice<'a, T> {
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

impl<'a, T: fmt::Debug> fmt::Debug for WeightedChoice<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("WeightedChoice")
         .field("items", &self.items)
         .field("weight_range", &self.weight_range)
         .finish()
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
fn ziggurat<R: Rng, P, Z>(rng: &mut R,
                          symmetric: bool,
                          x_tab: ziggurat_tables::ZigTable,
                          f_tab: ziggurat_tables::ZigTable,
                          mut pdf: P,
                          mut zero_case: Z)
                          -> f64
    where P: FnMut(f64) -> f64,
          Z: FnMut(&mut R, f64) -> f64
{
    const SCALE: f64 = (1u64 << 53) as f64;
    loop {
        // reimplement the f64 generation as an optimisation suggested
        // by the Doornik paper: we have a lot of precision-space
        // (i.e. there are 11 bits of the 64 of a u64 to use after
        // creating a f64), so we might as well reuse some to save
        // generating a whole extra random number. (Seems to be 15%
        // faster.)
        //
        // This unfortunately misses out on the benefits of direct
        // floating point generation if an RNG like dSMFT is
        // used. (That is, such RNGs create floats directly, highly
        // efficiently and overload next_f32/f64, so by not calling it
        // this may be slower than it would be otherwise.)
        // FIXME: investigate/optimise for the above.
        let bits: u64 = rng.gen();
        let i = (bits & 0xff) as usize;
        let f = (bits >> 11) as f64 / SCALE;

        // u is either U(-1, 1) or U(0, 1) depending on if this is a
        // symmetric distribution or not.
        let u = if symmetric { 2.0 * f - 1.0 } else { f };
        let x = u * x_tab[i];

        let test_x = if symmetric { x.abs() } else { x };

        // algebraically equivalent to |u| < x_tab[i+1]/x_tab[i] (or u < x_tab[i+1]/x_tab[i])
        if test_x < x_tab[i + 1] {
            return x;
        }
        if i == 0 {
            return zero_case(rng, u);
        }
        // algebraically equivalent to f1 + DRanU()*(f0 - f1) < 1
        if f_tab[i + 1] + (f_tab[i] - f_tab[i + 1]) * rng.gen::<f64>() < pdf(x) {
            return x;
        }
    }
}

#[cfg(test)]
mod tests {
    use {Rand, Rng};
    use super::{IndependentSample, RandSample, Sample, Weighted, WeightedChoice};

    #[derive(PartialEq, Debug)]
    struct ConstRand(usize);
    impl Rand for ConstRand {
        fn rand<R: Rng>(_: &mut R) -> ConstRand {
            ConstRand(0)
        }
    }

    // 0, 1, 2, 3, ...
    struct CountingRng {
        i: u32,
    }
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
        let mut rand_sample = RandSample::<ConstRand>::new();

        assert_eq!(rand_sample.sample(&mut ::test::rng()), ConstRand(0));
        assert_eq!(rand_sample.ind_sample(&mut ::test::rng()), ConstRand(0));
    }
    #[test]
    #[rustfmt_skip]
    fn test_weighted_choice() {
        // this makes assumptions about the internal implementation of
        // WeightedChoice, specifically: it doesn't reorder the items,
        // it doesn't do weird things to the RNG (so 0 maps to 0, 1 to
        // 1, internally; modulo a modulo operation).

        macro_rules! t {
            ($items:expr, $expected:expr) => {{
                let mut items = $items;
                let wc = WeightedChoice::new(&mut items);
                let expected = $expected;

                let mut rng = CountingRng { i: 0 };

                for &val in &expected {
                    assert_eq!(wc.ind_sample(&mut rng), val)
                }
            }}
        }

        t!(vec![Weighted { weight: 1, item: 10 }],
           [10]);

        // skip some
        t!(vec![Weighted { weight: 0, item: 20 },
                Weighted { weight: 2, item: 21 },
                Weighted { weight: 0, item: 22 },
                Weighted { weight: 1, item: 23 }],
           [21, 21, 23]);

        // different weights
        t!(vec![Weighted { weight: 4, item: 30 },
                Weighted { weight: 3, item: 31 }],
           [30, 30, 30, 30, 31, 31, 31]);

        // check that we're binary searching
        // correctly with some vectors of odd
        // length.
        t!(vec![Weighted { weight: 1, item: 40 },
                Weighted { weight: 1, item: 41 },
                Weighted { weight: 1, item: 42 },
                Weighted { weight: 1, item: 43 },
                Weighted { weight: 1, item: 44 }],
           [40, 41, 42, 43, 44]);
        t!(vec![Weighted { weight: 1, item: 50 },
                Weighted { weight: 1, item: 51 },
                Weighted { weight: 1, item: 52 },
                Weighted { weight: 1, item: 53 },
                Weighted { weight: 1, item: 54 },
                Weighted { weight: 1, item: 55 },
                Weighted { weight: 1, item: 56 }],
           [50, 51, 52, 53, 54, 55, 56]);
    }

    #[test]
    #[should_panic]
    fn test_weighted_choice_no_items() {
        WeightedChoice::<isize>::new(&mut []);
    }
    #[test]
    #[should_panic]
    #[rustfmt_skip]
    fn test_weighted_choice_zero_weight() {
        WeightedChoice::new(&mut [Weighted { weight: 0, item: 0 },
                                  Weighted { weight: 0, item: 1 }]);
    }
    #[test]
    #[should_panic]
    #[rustfmt_skip]
    fn test_weighted_choice_weight_overflows() {
        let x = (!0) as usize / 2; // x + x + 2 is the overflow
        WeightedChoice::new(&mut [Weighted { weight: x, item: 0 },
                                  Weighted { weight: 1, item: 1 },
                                  Weighted { weight: x, item: 2 },
                                  Weighted { weight: 1, item: 3 }]);
    }
}
