// FIXME(f16_f128): only tested on platforms that have symbols and aren't buggy
#![cfg(target_has_reliable_f128)]

#[cfg(any(miri, target_has_reliable_f128_math))]
use super::assert_approx_eq;
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

#[test]
#[cfg(any(miri, target_has_reliable_f128_math))]
fn test_max_recip() {
    assert_approx_eq!(
        f128::MAX.recip(),
        8.40525785778023376565669454330438228902076605e-4933,
        1e-4900
    );
}

#[test]
fn test_from() {
    assert_biteq!(f128::from(false), 0.0);
    assert_biteq!(f128::from(true), 1.0);
    assert_biteq!(f128::from(u8::MIN), 0.0);
    assert_biteq!(f128::from(42_u8), 42.0);
    assert_biteq!(f128::from(u8::MAX), 255.0);
    assert_biteq!(f128::from(i8::MIN), -128.0);
    assert_biteq!(f128::from(42_i8), 42.0);
    assert_biteq!(f128::from(i8::MAX), 127.0);
    assert_biteq!(f128::from(u16::MIN), 0.0);
    assert_biteq!(f128::from(42_u16), 42.0);
    assert_biteq!(f128::from(u16::MAX), 65535.0);
    assert_biteq!(f128::from(i16::MIN), -32768.0);
    assert_biteq!(f128::from(42_i16), 42.0);
    assert_biteq!(f128::from(i16::MAX), 32767.0);
    assert_biteq!(f128::from(u32::MIN), 0.0);
    assert_biteq!(f128::from(42_u32), 42.0);
    assert_biteq!(f128::from(u32::MAX), 4294967295.0);
    assert_biteq!(f128::from(i32::MIN), -2147483648.0);
    assert_biteq!(f128::from(42_i32), 42.0);
    assert_biteq!(f128::from(i32::MAX), 2147483647.0);
    // FIXME(f16_f128): Uncomment these tests once the From<{u64,i64}> impls are added.
    // assert_eq!(f128::from(u64::MIN), 0.0);
    // assert_eq!(f128::from(42_u64), 42.0);
    // assert_eq!(f128::from(u64::MAX), 18446744073709551615.0);
    // assert_eq!(f128::from(i64::MIN), -9223372036854775808.0);
    // assert_eq!(f128::from(42_i64), 42.0);
    // assert_eq!(f128::from(i64::MAX), 9223372036854775807.0);
}
