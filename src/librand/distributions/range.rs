// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Generating numbers between two others.

// this is surprisingly complicated to be both generic & correct

use std::num::Bounded;
use Rng;
use distributions::{Sample, IndependentSample};

/// Sample values uniformly between two bounds.
///
/// This gives a uniform distribution (assuming the RNG used to sample
/// it is itself uniform & the `SampleRange` implementation for the
/// given type is correct), even for edge cases like `low = 0u8`,
/// `high = 170u8`, for which a naive modulo operation would return
/// numbers less than 85 with double the probability to those greater
/// than 85.
///
/// Types should attempt to sample in `[low, high)`, i.e., not
/// including `high`, but this may be very difficult. All the
/// primitive integer types satisfy this property, and the float types
/// normally satisfy it, but rounding may mean `high` can occur.
///
/// # Example
///
/// ```rust
/// use rand::distributions::{IndependentSample, Range};
///
/// fn main() {
///     let between = Range::new(10u, 10000u);
///     let mut rng = rand::task_rng();
///     let mut sum = 0;
///     for _ in range(0, 1000) {
///         sum += between.ind_sample(&mut rng);
///     }
///     println!("{}", sum);
/// }
/// ```
pub struct Range<X> {
    low: X,
    range: X,
    accept_zone: X
}

impl<X: SampleRange + Ord> Range<X> {
    /// Create a new `Range` instance that samples uniformly from
    /// `[low, high)`. Fails if `low >= high`.
    pub fn new(low: X, high: X) -> Range<X> {
        assert!(low < high, "Range::new called with `low >= high`");
        SampleRange::construct_range(low, high)
    }
}

impl<Sup: SampleRange> Sample<Sup> for Range<Sup> {
    #[inline]
    fn sample<R: Rng>(&mut self, rng: &mut R) -> Sup { self.ind_sample(rng) }
}
impl<Sup: SampleRange> IndependentSample<Sup> for Range<Sup> {
    fn ind_sample<R: Rng>(&self, rng: &mut R) -> Sup {
        SampleRange::sample_range(self, rng)
    }
}

/// The helper trait for types that have a sensible way to sample
/// uniformly between two values. This should not be used directly,
/// and is only to facilitate `Range`.
pub trait SampleRange {
    /// Construct the `Range` object that `sample_range`
    /// requires. This should not ever be called directly, only via
    /// `Range::new`, which will check that `low < high`, so this
    /// function doesn't have to repeat the check.
    fn construct_range(low: Self, high: Self) -> Range<Self>;

    /// Sample a value from the given `Range` with the given `Rng` as
    /// a source of randomness.
    fn sample_range<R: Rng>(r: &Range<Self>, rng: &mut R) -> Self;
}

macro_rules! integer_impl {
    ($ty:ty, $unsigned:ty) => {
        impl SampleRange for $ty {
            // we play free and fast with unsigned vs signed here
            // (when $ty is signed), but that's fine, since the
            // contract of this macro is for $ty and $unsigned to be
            // "bit-equal", so casting between them is a no-op & a
            // bijection.

            fn construct_range(low: $ty, high: $ty) -> Range<$ty> {
                let range = high as $unsigned - low as $unsigned;
                let unsigned_max: $unsigned = Bounded::max_value();

                // this is the largest number that fits into $unsigned
                // that `range` divides evenly, so, if we've sampled
                // `n` uniformly from this region, then `n % range` is
                // uniform in [0, range)
                let zone = unsigned_max - unsigned_max % range;

                Range {
                    low: low,
                    range: range as $ty,
                    accept_zone: zone as $ty
                }
            }
            #[inline]
            fn sample_range<R: Rng>(r: &Range<$ty>, rng: &mut R) -> $ty {
                loop {
                    // rejection sample
                    let v = rng.gen::<$unsigned>();
                    // until we find something that fits into the
                    // region which r.range evenly divides (this will
                    // be uniformly distributed)
                    if v < r.accept_zone as $unsigned {
                        // and return it, with some adjustments
                        return r.low + (v % r.range as $unsigned) as $ty;
                    }
                }
            }
        }
    }
}

integer_impl! { i8, u8 }
integer_impl! { i16, u16 }
integer_impl! { i32, u32 }
integer_impl! { i64, u64 }
integer_impl! { int, uint }
integer_impl! { u8, u8 }
integer_impl! { u16, u16 }
integer_impl! { u32, u32 }
integer_impl! { u64, u64 }
integer_impl! { uint, uint }

macro_rules! float_impl {
    ($ty:ty) => {
        impl SampleRange for $ty {
            fn construct_range(low: $ty, high: $ty) -> Range<$ty> {
                Range {
                    low: low,
                    range: high - low,
                    accept_zone: 0.0 // unused
                }
            }
            fn sample_range<R: Rng>(r: &Range<$ty>, rng: &mut R) -> $ty {
                r.low + r.range * rng.gen()
            }
        }
    }
}

float_impl! { f32 }
float_impl! { f64 }

#[cfg(test)]
mod tests {
    use distributions::{Sample, IndependentSample};
    use {Rng, task_rng};
    use super::Range;
    use std::num::Bounded;

    #[should_fail]
    #[test]
    fn test_range_bad_limits_equal() {
        Range::new(10, 10);
    }
    #[should_fail]
    #[test]
    fn test_range_bad_limits_flipped() {
        Range::new(10, 5);
    }

    #[test]
    fn test_integers() {
        let mut rng = task_rng();
        macro_rules! t (
            ($($ty:ty),*) => {{
                $(
                   let v: &[($ty, $ty)] = [(0, 10),
                                           (10, 127),
                                           (Bounded::min_value(), Bounded::max_value())];
                   for &(low, high) in v.iter() {
                        let mut sampler: Range<$ty> = Range::new(low, high);
                        for _ in range(0, 1000) {
                            let v = sampler.sample(&mut rng);
                            assert!(low <= v && v < high);
                            let v = sampler.ind_sample(&mut rng);
                            assert!(low <= v && v < high);
                        }
                    }
                 )*
            }}
        );
        t!(i8, i16, i32, i64, int,
           u8, u16, u32, u64, uint)
    }

    #[test]
    fn test_floats() {
        let mut rng = task_rng();
        macro_rules! t (
            ($($ty:ty),*) => {{
                $(
                   let v: &[($ty, $ty)] = [(0.0, 100.0),
                                           (-1e35, -1e25),
                                           (1e-35, 1e-25),
                                           (-1e35, 1e35)];
                   for &(low, high) in v.iter() {
                        let mut sampler: Range<$ty> = Range::new(low, high);
                        for _ in range(0, 1000) {
                            let v = sampler.sample(&mut rng);
                            assert!(low <= v && v < high);
                            let v = sampler.ind_sample(&mut rng);
                            assert!(low <= v && v < high);
                        }
                    }
                 )*
            }}
        );

        t!(f32, f64)
    }

}
