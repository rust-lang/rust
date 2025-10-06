// FIXME(f16_f128): only tested on platforms that have symbols and aren't buggy
#![cfg(target_has_reliable_f16)]

use super::{assert_approx_eq, assert_biteq};

/// Tolerance for results on the order of 10.0e-2
#[allow(unused)]
const TOL_N2: f16 = 0.0001;

/// Tolerance for results on the order of 10.0e+0
#[allow(unused)]
const TOL_0: f16 = 0.01;

/// Tolerance for results on the order of 10.0e+2
#[allow(unused)]
const TOL_P2: f16 = 0.5;

/// Tolerance for results on the order of 10.0e+4
#[allow(unused)]
const TOL_P4: f16 = 10.0;

// FIXME(f16_f128,miri): many of these have to be disabled since miri does not yet support
// the intrinsics.

#[test]
#[cfg(any(miri, target_has_reliable_f16_math))]
fn test_max_recip() {
    assert_approx_eq!(f16::MAX.recip(), 1.526624e-5f16, 1e-4);
}

#[test]
fn test_from() {
    assert_biteq!(f16::from(false), 0.0);
    assert_biteq!(f16::from(true), 1.0);
    assert_biteq!(f16::from(u8::MIN), 0.0);
    assert_biteq!(f16::from(42_u8), 42.0);
    assert_biteq!(f16::from(u8::MAX), 255.0);
    assert_biteq!(f16::from(i8::MIN), -128.0);
    assert_biteq!(f16::from(42_i8), 42.0);
    assert_biteq!(f16::from(i8::MAX), 127.0);
}
