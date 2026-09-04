#[cfg(target_arch = "powerpc")]
use core::arch::powerpc::ppcf128;
#[cfg(target_arch = "powerpc64")]
use core::arch::powerpc64::ppcf128;
use std::assert_matches;

const _: () = assert!(size_of::<ppcf128>() == 16);
const _: () = assert!(align_of::<ppcf128>() == 16);

#[test]
fn constants() {
    assert_matches!(
        ppcf128::MIN.to_le_bytes(),
        [
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xef, 0xff, //
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x8f, 0xfc,
        ]
    );

    assert_eq!(
        ppcf128::MAX.to_le_bytes(),
        [
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xef, 0x7f, //
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x8f, 0x7c,
        ]
    );

    assert_eq!(
        ppcf128::NAN.to_le_bytes(),
        [
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f, //
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]
    );

    assert_eq!(
        ppcf128::INFINITY.to_le_bytes(),
        [
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x7f, //
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]
    );

    assert_eq!(
        ppcf128::NEG_INFINITY.to_le_bytes(),
        [
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0xff, //
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]
    );
}

#[test]
fn impls() {
    const ONE: ppcf128 = ppcf128::from_components(1.0, 0.0);
    const TWO: ppcf128 = ppcf128::from_components(2.0, 0.0);

    assert_eq!(ONE, ONE);
    assert_ne!(ONE, TWO);

    assert!(ONE <= ONE);
    assert!(ONE < TWO);
    assert!(TWO > ONE);
    assert!(TWO >= TWO);

    assert!(ONE < ppcf128::INFINITY);
    assert!(ONE > ppcf128::NEG_INFINITY);

    assert_ne!(ONE, ppcf128::NAN);
    assert_ne!(ppcf128::NAN, ppcf128::NAN);

    assert_eq!(ppcf128::default().to_components(), (0.0, 0.0));
}

macro_rules! assert_eq_normalized {
    (($large:expr, $small:expr) ==> ($expected_large:expr, $expected_small:expr)) => {{
        let (large, small) = ppcf128::from_components($large, $small).to_components();
        let (expected_large, expected_small): (f64, f64) = ($expected_large, $expected_small);

        if expected_large.is_nan() {
            assert!(large.is_nan(), "expected NaN, got {large}");
        } else {
            assert_eq!(large, expected_large);
        }

        if expected_small.is_nan() {
            assert!(small.is_nan(), "expected NaN, got {small}");
        } else {
            assert_eq!(small, expected_small);
        }
    }};
}

#[test]
fn normalize() {
    // Already normalized.
    assert_eq_normalized!((1.0, 0.0) ==> (1.0, 0.0));
    assert_eq_normalized!((-1.0, 0.0) ==> (-1.0, 0.0));

    // The value is normalized.
    assert_eq_normalized!((1.0, 0.5) ==> (1.5, 0.0));
    assert_eq_normalized!((-1.0, 0.5) ==> (-0.5, 0.0));
    assert_eq_normalized!((0.0, 1.0) ==> (1.0, 0.0));
    assert_eq_normalized!((0.0, -1.0) ==> (-1.0, 0.0));

    let large = 2.0f64.powi(53);
    assert_eq_normalized!((large, -(large - 1.0)) ==> (1.0, 0.0));

    // Exact cancellation.
    assert_eq_normalized!((1.0, -1.0) ==> (0.0, 0.0));
    assert_eq_normalized!((-1.0, 1.0) ==> (0.0, 0.0));

    // A component too small to affect the high part remains in the low part.
    let half_ulp = f64::EPSILON / 2.0;
    assert_eq_normalized!((1.0, half_ulp) ==> (1.0, half_ulp));
    assert_eq_normalized!((-1.0, -half_ulp) ==> (-1.0, -half_ulp));

    assert_eq_normalized!((1.0, -half_ulp) ==> (1.0 - half_ulp, 0.0));
    assert_eq_normalized!((-1.0, half_ulp) ==> (-1.0 + half_ulp, 0.0));

    // Rounding the sum produces a compensating low component.
    let three_quarters_ulp = 3.0 * f64::EPSILON / 4.0;
    assert_eq_normalized!(
        (1.0, three_quarters_ulp) ==>
        (1.0 + f64::EPSILON, -f64::EPSILON / 4.0)
    );

    // Normalization works across a large difference in exponent.
    let large = 2.0f64.powi(100);
    assert_eq_normalized!((1.0, large) ==> (large, 1.0));

    // Infinity with a zero low component is already normalized.
    assert_eq_normalized!((f64::INFINITY, 0.0) ==> (f64::INFINITY, 0.0));
    assert_eq_normalized!((f64::NEG_INFINITY, 0.0) ==> (f64::NEG_INFINITY, 0.0));

    cfg_select! {
        target_endian = "little" => {
            // Covers powerpc64le (elfv2)
            assert_eq_normalized!((f64::INFINITY, 1.0) ==> (f64::INFINITY, 0.0));
            assert_eq_normalized!((f64::NEG_INFINITY, 1.0) ==> (f64::NEG_INFINITY, 0.0));
        }
        target_endian = "big" => {
            // Covers powerpc64 (elfv1), powerpc and aix
            assert_eq_normalized!((f64::INFINITY, 1.0) ==> (f64::INFINITY, 1.0));
            assert_eq_normalized!((f64::NEG_INFINITY, 1.0) ==> (f64::NEG_INFINITY, 1.0));
        }
    }

    // A finite high component combined with infinity normalizes to infinity.
    assert_eq_normalized!((1.0, f64::INFINITY) ==> (f64::INFINITY, 0.0));
    assert_eq_normalized!((1.0, f64::NEG_INFINITY) ==> (f64::NEG_INFINITY, 0.0));

    // Opposite infinities produce NaN.
    assert_eq_normalized!(
        (f64::INFINITY, f64::NEG_INFINITY) ==>
        (f64::NAN, 0.0)
    );

    assert_eq_normalized!((1.0, f64::NAN) ==> (f64::NAN, 0.0));
    assert_eq_normalized!((f64::NAN, 1.0) ==> (f64::NAN, 1.0));

    assert_matches!(ppcf128::checked_from_components(1.0, 0.0), Some(_));
    assert_matches!(ppcf128::checked_from_components(0.0, 1.0), None);
}
