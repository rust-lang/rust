use core::num::flt2dec::estimator::*;

#[test]
fn test_estimate_scaling_factor() {
    macro_rules! assert_almost_eq {
        ($actual:expr, $expected:expr) => {{
            let actual = $actual;
            let expected = $expected;
            println!(
                "{} - {} = {} - {} = {}",
                stringify!($expected),
                stringify!($actual),
                expected,
                actual,
                expected - actual
            );
            assert!(
                expected == actual || expected == actual + 1,
                "expected {}, actual {}",
                expected,
                actual
            );
        }};
    }

    assert_almost_eq!(estimate_scaling_factor(1, 0), 0);
    assert_almost_eq!(estimate_scaling_factor(2, 0), 1);
    assert_almost_eq!(estimate_scaling_factor(10, 0), 1);
    assert_almost_eq!(estimate_scaling_factor(11, 0), 2);
    assert_almost_eq!(estimate_scaling_factor(100, 0), 2);
    assert_almost_eq!(estimate_scaling_factor(101, 0), 3);
    assert_almost_eq!(estimate_scaling_factor(10000000000000000000, 0), 19);
    assert_almost_eq!(estimate_scaling_factor(10000000000000000001, 0), 20);

    // 1/2^20 = 0.00000095367...
    assert_almost_eq!(estimate_scaling_factor(1 * 1048576 / 1000000, -20), -6);
    assert_almost_eq!(estimate_scaling_factor(1 * 1048576 / 1000000 + 1, -20), -5);
    assert_almost_eq!(estimate_scaling_factor(10 * 1048576 / 1000000, -20), -5);
    assert_almost_eq!(estimate_scaling_factor(10 * 1048576 / 1000000 + 1, -20), -4);
    assert_almost_eq!(estimate_scaling_factor(100 * 1048576 / 1000000, -20), -4);
    assert_almost_eq!(estimate_scaling_factor(100 * 1048576 / 1000000 + 1, -20), -3);
    assert_almost_eq!(estimate_scaling_factor(1048575, -20), 0);
    assert_almost_eq!(estimate_scaling_factor(1048576, -20), 0);
    assert_almost_eq!(estimate_scaling_factor(1048577, -20), 1);
    assert_almost_eq!(estimate_scaling_factor(10485759999999999999, -20), 13);
    assert_almost_eq!(estimate_scaling_factor(10485760000000000000, -20), 13);
    assert_almost_eq!(estimate_scaling_factor(10485760000000000001, -20), 14);

    // extreme values:
    // 2^-1074 = 4.94065... * 10^-324
    // (2^53-1) * 2^971 = 1.79763... * 10^308
    assert_almost_eq!(estimate_scaling_factor(1, -1074), -323);
    assert_almost_eq!(estimate_scaling_factor(0x1fffffffffffff, 971), 309);

    // Miri is too slow
    let step = if cfg!(miri) { 37 } else { 1 };

    for i in (-1074..972).step_by(step) {
        let expected = super::ldexp_f64(1.0, i).log10().ceil();
        assert_almost_eq!(estimate_scaling_factor(1, i as i16), expected as i16);
    }
}
