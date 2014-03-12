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

use std::num::Float;
use {Rng, Rand};
use distributions::{ziggurat, ziggurat_tables, Sample, IndependentSample};

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
/// ```rust
/// use rand::distributions::{Exp, IndependentSample};
///
/// let exp = Exp::new(2.0);
/// let v = exp.ind_sample(&mut rand::task_rng());
/// println!("{} is from a Exp(2) distribution", v);
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
        let Exp1(n) = rng.gen::<Exp1>();
        n * self.lambda_inverse
    }
}

#[cfg(test)]
mod test {
    use distributions::{Sample, IndependentSample};
    use {Rng, task_rng};
    use super::Exp;

    #[test]
    fn test_exp() {
        let mut exp = Exp::new(10.0);
        let mut rng = task_rng();
        for _ in range(0, 1000) {
            assert!(exp.sample(&mut rng) >= 0.0);
            assert!(exp.ind_sample(&mut rng) >= 0.0);
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
}

#[cfg(test)]
mod bench {
    extern crate test;
    use self::test::BenchHarness;
    use std::mem::size_of;
    use {XorShiftRng, RAND_BENCH_N};
    use super::Exp;
    use distributions::Sample;

    #[bench]
    fn rand_exp(bh: &mut BenchHarness) {
        let mut rng = XorShiftRng::new();
        let mut exp = Exp::new(2.71828 * 3.14159);

        bh.iter(|| {
            for _ in range(0, RAND_BENCH_N) {
                exp.sample(&mut rng);
            }
        });
        bh.bytes = size_of::<f64>() as u64 * RAND_BENCH_N;
    }
}
