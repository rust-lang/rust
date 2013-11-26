// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The Gamma distribution.

use rand::{Rng, Open01};
use super::{IndependentSample, Sample, StandardNormal, Exp};
use num;

/// The Gamma distribution `Gamma(shape, scale)` distribution.
///
/// The density function of this distribution is
///
/// ```
/// f(x) =  x^(k - 1) * exp(-x / θ) / (Γ(k) * θ^k)
/// ```
///
/// where `Γ` is the Gamma function, `k` is the shape and `θ` is the
/// scale and both `k` and `θ` are strictly positive.
///
/// The algorithm used is that described by Marsaglia & Tsang 2000[1],
/// falling back to directly sampling from an Exponential for `shape
/// == 1`, and using the boosting technique described in [1] for
/// `shape < 1`.
///
/// # Example
///
/// ```rust
/// use std::rand;
/// use std::rand::distributions::{IndependentSample, Gamma};
///
/// fn main() {
///     let gamma = Gamma::new(2.0, 5.0);
///     let v = gamma.ind_sample(rand::task_rng());
///     println!("{} is from a Gamma(2, 5) distribution", v);
/// }
/// ```
///
/// [1]: George Marsaglia and Wai Wan Tsang. 2000. "A Simple Method
/// for Generating Gamma Variables" *ACM Trans. Math. Softw.* 26, 3
/// (September 2000),
/// 363-372. DOI:[10.1145/358407.358414](http://doi.acm.org/10.1145/358407.358414)
pub enum Gamma {
    priv Large(GammaLargeShape),
    priv One(Exp),
    priv Small(GammaSmallShape)
}

// These two helpers could be made public, but saving the
// match-on-Gamma-enum branch from using them directly (e.g. if one
// knows that the shape is always > 1) doesn't appear to be much
// faster.

/// Gamma distribution where the shape parameter is less than 1.
///
/// Note, samples from this require a compulsory floating-point `pow`
/// call, which makes it significantly slower than sampling from a
/// gamma distribution where the shape parameter is greater than or
/// equal to 1.
///
/// See `Gamma` for sampling from a Gamma distribution with general
/// shape parameters.
struct GammaSmallShape {
    inv_shape: f64,
    large_shape: GammaLargeShape
}

/// Gamma distribution where the shape parameter is larger than 1.
///
/// See `Gamma` for sampling from a Gamma distribution with general
/// shape parameters.
struct GammaLargeShape {
    shape: f64,
    scale: f64,
    c: f64,
    d: f64
}

impl Gamma {
    /// Construct an object representing the `Gamma(shape, scale)`
    /// distribution.
    ///
    /// Fails if `shape <= 0` or `scale <= 0`.
    pub fn new(shape: f64, scale: f64) -> Gamma {
        assert!(shape > 0.0, "Gamma::new called with shape <= 0");
        assert!(scale > 0.0, "Gamma::new called with scale <= 0");

        match shape {
            1.0        => One(Exp::new(1.0 / scale)),
            0.0 .. 1.0 => Small(GammaSmallShape::new_raw(shape, scale)),
            _          => Large(GammaLargeShape::new_raw(shape, scale))
        }
    }
}

impl GammaSmallShape {
    fn new_raw(shape: f64, scale: f64) -> GammaSmallShape {
        GammaSmallShape {
            inv_shape: 1. / shape,
            large_shape: GammaLargeShape::new_raw(shape + 1.0, scale)
        }
    }
}

impl GammaLargeShape {
    fn new_raw(shape: f64, scale: f64) -> GammaLargeShape {
        let d = shape - 1. / 3.;
        GammaLargeShape {
            shape: shape,
            scale: scale,
            c: 1. / num::sqrt(9. * d),
            d: d
        }
    }
}

impl Sample<f64> for Gamma {
    fn sample<R: Rng>(&mut self, rng: &mut R) -> f64 { self.ind_sample(rng) }
}
impl Sample<f64> for GammaSmallShape {
    fn sample<R: Rng>(&mut self, rng: &mut R) -> f64 { self.ind_sample(rng) }
}
impl Sample<f64> for GammaLargeShape {
    fn sample<R: Rng>(&mut self, rng: &mut R) -> f64 { self.ind_sample(rng) }
}

impl IndependentSample<f64> for Gamma {
    fn ind_sample<R: Rng>(&self, rng: &mut R) -> f64 {
        match *self {
            Small(ref g) => g.ind_sample(rng),
            One(ref g) => g.ind_sample(rng),
            Large(ref g) => g.ind_sample(rng),
        }
    }
}
impl IndependentSample<f64> for GammaSmallShape {
    fn ind_sample<R: Rng>(&self, rng: &mut R) -> f64 {
        let u = *rng.gen::<Open01<f64>>();

        self.large_shape.ind_sample(rng) * num::pow(u, self.inv_shape)
    }
}
impl IndependentSample<f64> for GammaLargeShape {
    fn ind_sample<R: Rng>(&self, rng: &mut R) -> f64 {
        loop {
            let x = *rng.gen::<StandardNormal>();
            let v_cbrt = 1.0 + self.c * x;
            if v_cbrt <= 0.0 { // a^3 <= 0 iff a <= 0
                continue
            }

            let v = v_cbrt * v_cbrt * v_cbrt;
            let u = *rng.gen::<Open01<f64>>();

            let x_sqr = x * x;
            if u < 1.0 - 0.0331 * x_sqr * x_sqr ||
                num::ln(u) < 0.5 * x_sqr + self.d * (1.0 - v + num::ln(v)) {
                return self.d * v * self.scale
            }
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use mem::size_of;
    use rand::distributions::IndependentSample;
    use rand::{StdRng, RAND_BENCH_N};
    use extra::test::BenchHarness;
    use iter::range;
    use option::{Some, None};


    #[bench]
    fn bench_gamma_large_shape(bh: &mut BenchHarness) {
        let gamma = Gamma::new(10., 1.0);
        let mut rng = StdRng::new();

        bh.iter(|| {
            for _ in range(0, RAND_BENCH_N) {
                gamma.ind_sample(&mut rng);
            }
        });
        bh.bytes = size_of::<f64>() as u64 * RAND_BENCH_N;
    }

    #[bench]
    fn bench_gamma_small_shape(bh: &mut BenchHarness) {
        let gamma = Gamma::new(0.1, 1.0);
        let mut rng = StdRng::new();

        bh.iter(|| {
            for _ in range(0, RAND_BENCH_N) {
                gamma.ind_sample(&mut rng);
            }
        });
        bh.bytes = size_of::<f64>() as u64 * RAND_BENCH_N;
    }
}
