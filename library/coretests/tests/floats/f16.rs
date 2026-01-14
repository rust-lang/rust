// FIXME(f16_f128): only tested on platforms that have symbols and aren't buggy
#![cfg(target_has_reliable_f16)]

use super::assert_biteq;

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

// NOTE: test_from has been moved to mod.rs using float_test! macro
// See: from_bool, from_u8, from_i8 tests
