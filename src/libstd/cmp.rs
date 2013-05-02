// Copyright 2012-2013 The Rust Project Developers. See the
// COPYRIGHT file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Additional general-purpose comparison functionality.

use core::f32;
use core::f64;
use core::float;

pub static FUZZY_EPSILON: float = 1.0e-6;

pub trait FuzzyEq<Eps> {
    fn fuzzy_eq(&self, other: &Self) -> bool;
    fn fuzzy_eq_eps(&self, other: &Self, epsilon: &Eps) -> bool;
}

impl FuzzyEq<float> for float {
    fn fuzzy_eq(&self, other: &float) -> bool {
        self.fuzzy_eq_eps(other, &FUZZY_EPSILON)
    }

    fn fuzzy_eq_eps(&self, other: &float, epsilon: &float) -> bool {
        float::abs(*self - *other) < *epsilon
    }
}

impl FuzzyEq<f32> for f32 {
    fn fuzzy_eq(&self, other: &f32) -> bool {
        self.fuzzy_eq_eps(other, &(FUZZY_EPSILON as f32))
    }

    fn fuzzy_eq_eps(&self, other: &f32, epsilon: &f32) -> bool {
        f32::abs(*self - *other) < *epsilon
    }
}

impl FuzzyEq<f64> for f64 {
    fn fuzzy_eq(&self, other: &f64) -> bool {
        self.fuzzy_eq_eps(other, &(FUZZY_EPSILON as f64))
    }

    fn fuzzy_eq_eps(&self, other: &f64, epsilon: &f64) -> bool {
        f64::abs(*self - *other) < *epsilon
    }
}

#[test]
fn test_fuzzy_equals() {
    assert!((&1.0f).fuzzy_eq(&1.0));
    assert!((&1.0f32).fuzzy_eq(&1.0f32));
    assert!((&1.0f64).fuzzy_eq(&1.0f64));
}

#[test]
fn test_fuzzy_eq_eps() {
    assert!((&1.2f).fuzzy_eq_eps(&0.9, &0.5));
    assert!(!(&1.5f).fuzzy_eq_eps(&0.9, &0.5));
}

#[cfg(test)]
mod test_complex{
    use cmp::*;

    struct Complex { r: float, i: float }

    impl FuzzyEq<float> for Complex {
        fn fuzzy_eq(&self, other: &Complex) -> bool {
            self.fuzzy_eq_eps(other, &FUZZY_EPSILON)
        }

        fn fuzzy_eq_eps(&self, other: &Complex,
                             epsilon: &float) -> bool {
            self.r.fuzzy_eq_eps(&other.r, epsilon) &&
            self.i.fuzzy_eq_eps(&other.i, epsilon)
        }
    }

    #[test]
    fn test_fuzzy_equals() {
        let a = Complex {r: 0.9, i: 0.9};
        let b = Complex {r: 0.9, i: 0.9};

        assert!((a.fuzzy_eq(&b)));
    }

    #[test]
    fn test_fuzzy_eq_eps() {
        let other = Complex {r: 0.9, i: 0.9};

        assert!((&Complex {r: 0.9, i: 1.2}).fuzzy_eq_eps(&other, &0.5));
        assert!((&Complex {r: 1.2, i: 0.9}).fuzzy_eq_eps(&other, &0.5));
        assert!(!(&Complex {r: 0.9, i: 1.5}).fuzzy_eq_eps(&other, &0.5));
        assert!(!(&Complex {r: 1.5, i: 0.9}).fuzzy_eq_eps(&other, &0.5));
    }
}
