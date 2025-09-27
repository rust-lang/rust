use core::num::flt2dec::estimator::*;

use crate::num::ldexp_f64;

#[test]
fn test_estimate_scaling_factor() {
    fn assert_estimate(mant: u64, exp: isize, want: isize) {
        let got = estimate_scaling_factor(mant, exp);
        let tolerate = want - 1;
        assert!(
            got == want || got == tolerate,
            "mant {mant} and exp {exp} got {got}, want {want} or tolerate {tolerate}",
        );
    }

    assert_estimate(1, 0, 0);
    assert_estimate(2, 0, 1);
    assert_estimate(10, 0, 1);
    assert_estimate(11, 0, 2);
    assert_estimate(100, 0, 2);
    assert_estimate(101, 0, 3);
    assert_estimate(10000000000000000000, 0, 19);
    assert_estimate(10000000000000000001, 0, 20);

    // 1/2^20 = 0.00000095367...
    assert_estimate(1 * 1048576 / 1000000, -20, -6);
    assert_estimate(1 * 1048576 / 1000000 + 1, -20, -5);
    assert_estimate(10 * 1048576 / 1000000, -20, -5);
    assert_estimate(10 * 1048576 / 1000000 + 1, -20, -4);
    assert_estimate(100 * 1048576 / 1000000, -20, -4);
    assert_estimate(100 * 1048576 / 1000000 + 1, -20, -3);
    assert_estimate(1048575, -20, 0);
    assert_estimate(1048576, -20, 0);
    assert_estimate(1048577, -20, 1);
    assert_estimate(10485759999999999999, -20, 13);
    assert_estimate(10485760000000000000, -20, 13);
    assert_estimate(10485760000000000001, -20, 14);

    // extreme values:
    // 2^-1074 = 4.94065... * 10^-324
    // (2^53-1) * 2^971 = 1.79763... * 10^308
    assert_estimate(1, -1074, -323);
    assert_estimate(0x1fffffffffffff, 971, 309);

    // Miri is too slow
    let step = if cfg!(miri) { 37 } else { 1 };

    for i in (-1074..972).step_by(step) {
        let want = ldexp_f64(1.0, i).log10().ceil();
        assert_estimate(1, i as isize, want as isize);
    }
}
