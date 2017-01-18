// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The exponential distribution.

use core::fmt;

#[cfg(not(test))] // only necessary for no_std
use FloatMath;

use {Rand, Rng};
use distributions::{IndependentSample, Sample, ziggurat, ziggurat_tables};

/// A wrapper around an `f64` to generate Exp(1) random numbers.
///
/// See `Exp` for the general exponential distribution. Note that this has to
/// be unwrapped before use as an `f64` (using either `*` or `mem::transmute`
/// is safe).
///
/// Implemented via the ZIGNOR variant[1] of the Ziggurat method. The
/// exact description in the paper was adjusted to use tables for the
/// exponential distribution rather than normal.
///
/// [1]: Jurgen A. Doornik (2005). [*An Improved Ziggurat Method to
/// Generate Normal Random
/// Samples*](http://www.doornik.com/research/ziggurat.pdf). Nuffield
/// College, Oxford
#[derive(Copy, Clone)]
pub struct Exp1(pub f64);

// This could be done via `-rng.gen::<f64>().ln()` but that is slower.
impl Rand for Exp1 {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> Exp1 {
        #[inline]
        fn pdf(x: f64) -> f64 {
            (-x).exp()
        }
        #[inline]
        fn zero_case<R: Rng>(rng: &mut R, _u: f64) -> f64 {
            ziggurat_tables::ZIG_EXP_R - rng.gen::<f64>().ln()
        }

        Exp1(ziggurat(rng,
                      false,
                      &ziggurat_tables::ZIG_EXP_X,
                      &ziggurat_tables::ZIG_EXP_F,
                      pdf,
                      zero_case))
    }
}

impl fmt::Debug for Exp1 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Exp1")
         .field(&self.0)
         .finish()
    }
}

/// The exponential distribution `Exp(lambda)`.
///
/// This distribution has density function: `f(x) = lambda *
/// exp(-lambda * x)` for `x > 0`.
#[derive(Copy, Clone)]
pub struct Exp {
    /// `lambda` stored as `1/lambda`, since this is what we scale by.
    lambda_inverse: f64,
}

impl Exp {
    /// Construct a new `Exp` with the given shape parameter
    /// `lambda`. Panics if `lambda <= 0`.
    pub fn new(lambda: f64) -> Exp {
        assert!(lambda > 0.0, "Exp::new called with `lambda` <= 0");
        Exp { lambda_inverse: 1.0 / lambda }
    }
}

impl Sample<f64> for Exp {
    fn sample<R: Rng>(&mut self, rng: &mut R) -> f64 {
        self.ind_sample(rng)
    }
}

impl IndependentSample<f64> for Exp {
    fn ind_sample<R: Rng>(&self, rng: &mut R) -> f64 {
        let Exp1(n) = rng.gen::<Exp1>();
        n * self.lambda_inverse
    }
}

impl fmt::Debug for Exp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Exp")
         .field("lambda_inverse", &self.lambda_inverse)
         .finish()
    }
}

#[cfg(test)]
mod tests {
    use distributions::{IndependentSample, Sample};
    use super::Exp;

    #[test]
    fn test_exp() {
        let mut exp = Exp::new(10.0);
        let mut rng = ::test::rng();
        for _ in 0..1000 {
            assert!(exp.sample(&mut rng) >= 0.0);
            assert!(exp.ind_sample(&mut rng) >= 0.0);
        }
    }
    #[test]
    #[should_panic]
    fn test_exp_invalid_lambda_zero() {
        Exp::new(0.0);
    }
    #[test]
    #[should_panic]
    fn test_exp_invalid_lambda_neg() {
        Exp::new(-10.0);
    }
}

#[cfg(test)]
mod bench {
    extern crate test;

    use self::test::Bencher;
    use std::mem::size_of;
    use super::Exp;
    use distributions::Sample;

    #[bench]
    fn rand_exp(b: &mut Bencher) {
        let mut rng = ::test::weak_rng();
        let mut exp = Exp::new(2.71828 * 3.14159);

        b.iter(|| {
            for _ in 0..::RAND_BENCH_N {
                exp.sample(&mut rng);
            }
        });
        b.bytes = size_of::<f64>() as u64 * ::RAND_BENCH_N;
    }
}
