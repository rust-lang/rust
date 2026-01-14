// FIXME(f16_f128): only tested on platforms that have symbols and aren't buggy
#![cfg(target_has_reliable_f128)]

use super::assert_biteq;

// Note these tolerances make sense around zero, but not for more extreme exponents.

/// Default tolerances. Works for values that should be near precise but not exact. Roughly
/// the precision carried by `100 * 100`.
#[allow(unused)]
const TOL: f128 = 1e-12;

/// For operations that are near exact, usually not involving math of different
/// signs.
#[allow(unused)]
const TOL_PRECISE: f128 = 1e-28;

// FIXME(f16_f128,miri): many of these have to be disabled since miri does not yet support
// the intrinsics.

// NOTE: test_from has been moved to mod.rs using float_test! macro
// See: from_bool, from_u8, from_i8, from_u16, from_i16, from_u32, from_i32 tests
