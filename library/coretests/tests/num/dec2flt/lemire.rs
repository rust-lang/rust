use core::num::dec2flt::float::RawFloat;
use core::num::dec2flt::lemire::compute_float;

fn compute_float32(q: i64, w: u64) -> (i32, u64) {
    let fp = compute_float::<f32>(q, w);
    (fp.p_biased, fp.m)
}

fn compute_float64(q: i64, w: u64) -> (i32, u64) {
    let fp = compute_float::<f64>(q, w);
    (fp.p_biased, fp.m)
}

#[test]
fn compute_float_f32_rounding() {
    // These test near-halfway cases for single-precision floats.
    assert_eq!(compute_float32(0, 16777216), (151, 0));
    assert_eq!(compute_float32(0, 16777217), (151, 0));
    assert_eq!(compute_float32(0, 16777218), (151, 1));
    assert_eq!(compute_float32(0, 16777219), (151, 2));
    assert_eq!(compute_float32(0, 16777220), (151, 2));

    // These are examples of the above tests, with
    // digits from the exponent shifted to the mantissa.
    assert_eq!(compute_float32(-10, 167772160000000000), (151, 0));
    assert_eq!(compute_float32(-10, 167772170000000000), (151, 0));
    assert_eq!(compute_float32(-10, 167772180000000000), (151, 1));
    // Let's check the lines to see if anything is different in table...
    assert_eq!(compute_float32(-10, 167772190000000000), (151, 2));
    assert_eq!(compute_float32(-10, 167772200000000000), (151, 2));

    // Check the rounding point between infinity and the next representable number down
    assert_eq!(compute_float32(38, 3), (f32::INFINITE_POWER - 1, 6402534));
    assert_eq!(compute_float32(38, 4), (f32::INFINITE_POWER, 0)); // infinity
    assert_eq!(compute_float32(20, 3402823470000000000), (f32::INFINITE_POWER - 1, 8388607));
}

#[test]
fn compute_float_f64_rounding() {
    // These test near-halfway cases for double-precision floats.
    assert_eq!(compute_float64(0, 9007199254740992), (1076, 0));
    assert_eq!(compute_float64(0, 9007199254740993), (1076, 0));
    assert_eq!(compute_float64(0, 9007199254740994), (1076, 1));
    assert_eq!(compute_float64(0, 9007199254740995), (1076, 2));
    assert_eq!(compute_float64(0, 9007199254740996), (1076, 2));
    assert_eq!(compute_float64(0, 18014398509481984), (1077, 0));
    assert_eq!(compute_float64(0, 18014398509481986), (1077, 0));
    assert_eq!(compute_float64(0, 18014398509481988), (1077, 1));
    assert_eq!(compute_float64(0, 18014398509481990), (1077, 2));
    assert_eq!(compute_float64(0, 18014398509481992), (1077, 2));

    // These are examples of the above tests, with
    // digits from the exponent shifted to the mantissa.
    assert_eq!(compute_float64(-3, 9007199254740992000), (1076, 0));
    assert_eq!(compute_float64(-3, 9007199254740993000), (1076, 0));
    assert_eq!(compute_float64(-3, 9007199254740994000), (1076, 1));
    assert_eq!(compute_float64(-3, 9007199254740995000), (1076, 2));
    assert_eq!(compute_float64(-3, 9007199254740996000), (1076, 2));
}
