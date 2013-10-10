// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Sampling from random distributions

// Some implementations use the Ziggurat method
// https://en.wikipedia.org/wiki/Ziggurat_algorithm
//
// The version used here is ZIGNOR [Doornik 2005, "An Improved
// Ziggurat Method to Generate Normal Random Samples"] which is slower
// (about double, it generates an extra random number) than the
// canonical version [Marsaglia & Tsang 2000, "The Ziggurat Method for
// Generating Random Variables"], but more robust. If one wanted, one
// could implement VIZIGNOR the ZIGNOR paper for more speed.

use num;
use rand::{Rng,Rand};

pub use self::range::Range;

pub mod range;

/// Things that can be used to create a random instance of `Support`.
pub trait Sample<Support> {
    /// Generate a random value of `Support`, using `rng` as the
    /// source of randomness.
    fn sample<R: Rng>(&mut self, rng: &mut R) -> Support;
}

/// `Sample`s that do not require keeping track of state, so each
/// sample is (statistically) independent of all others, assuming the
/// `Rng` used has this property.
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

mod ziggurat_tables;

// inlining should mean there is no performance penalty for this
#[inline]
fn ziggurat<R:Rng>(rng: &mut R,
                   center_u: bool,
                   X: ziggurat_tables::ZigTable,
                   F: ziggurat_tables::ZigTable,
                   F_DIFF: ziggurat_tables::ZigTable,
                   pdf: &'static fn(f64) -> f64, // probability density function
                   zero_case: &'static fn(&mut R, f64) -> f64) -> f64 {
    loop {
        let u = if center_u {2.0 * rng.gen() - 1.0} else {rng.gen()};
        let i: uint = rng.gen::<uint>() & 0xff;
        let x = u * X[i];

        let test_x = if center_u {num::abs(x)} else {x};

        // algebraically equivalent to |u| < X[i+1]/X[i] (or u < X[i+1]/X[i])
        if test_x < X[i + 1] {
            return x;
        }
        if i == 0 {
            return zero_case(rng, u);
        }
        // algebraically equivalent to f1 + DRanU()*(f0 - f1) < 1
        if F[i+1] + F_DIFF[i+1] * rng.gen() < pdf(x) {
            return x;
        }
    }
}

/// A wrapper around an `f64` to generate N(0, 1) random numbers (a.k.a.  a
/// standard normal, or Gaussian). Multiplying the generated values by the
/// desired standard deviation `sigma` then adding the desired mean `mu` will
/// give N(mu, sigma^2) distributed random numbers.
///
/// Note that this has to be unwrapped before use as an `f64` (using either
/// `*` or `cast::transmute` is safe).
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
            &ziggurat_tables::ZIG_NORM_F, &ziggurat_tables::ZIG_NORM_F_DIFF,
            pdf, zero_case))
    }
}

/// The `N(mean, std_dev**2)` distribution, i.e. samples from a normal
/// distribution with mean `mean` and standard deviation `std_dev`.
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

/// A wrapper around an `f64` to generate Exp(1) random numbers. Dividing by
/// the desired rate `lambda` will give Exp(lambda) distributed random
/// numbers.
///
/// Note that this has to be unwrapped before use as an `f64` (using either
/// `*` or `cast::transmute` is safe).
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
                      &ziggurat_tables::ZIG_EXP_F, &ziggurat_tables::ZIG_EXP_F_DIFF,
                      pdf, zero_case))
    }
}

/// The `Exp(lambda)` distribution; i.e. samples from the exponential
/// distribution with rate parameter `lambda`.
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
}
