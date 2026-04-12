//! Verify that matching against a constant array pattern produces correct
//! results at runtime, complementing the MIR test in
//! `tests/mir-opt/building/match/aggregate_array_eq.rs` which checks that
//! a single aggregate `PartialEq::eq` call is emitted.
//@ run-pass

fn array_match(x: [u8; 4]) -> bool {
    matches!(x, [1, 2, 3, 4])
}

fn main() {
    assert!(array_match([1, 2, 3, 4]));
    assert!(!array_match([1, 2, 3, 5]));
    assert!(!array_match([0, 0, 0, 0]));
    assert!(!array_match([4, 3, 2, 1]));
}
