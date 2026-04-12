//! Verify that matching against a constant array pattern produces correct
//! results at runtime, complementing the MIR test in
//! `tests/mir-opt/building/match/aggregate_array_eq.rs` which checks that
//! a single aggregate `PartialEq::eq` call is emitted.
//@ run-pass

fn array_match(x: [u8; 4]) -> bool {
    matches!(x, [1, 2, 3, 4])
}

#[derive(Debug, PartialEq)]
enum MyEnum {
    A,
    B,
    C,
    D,
}

// Regression test for https://github.com/rust-lang/rust/issues/103073.
fn try_from_matched(value: [u8; 4]) -> Result<MyEnum, ()> {
    match &value {
        b"ABCD" => Ok(MyEnum::A),
        b"EFGH" => Ok(MyEnum::B),
        b"IJKL" => Ok(MyEnum::C),
        b"MNOP" => Ok(MyEnum::D),
        _ => Err(()),
    }
}

fn main() {
    assert!(array_match([1, 2, 3, 4]));
    assert!(!array_match([1, 2, 3, 5]));
    assert!(!array_match([0, 0, 0, 0]));
    assert!(!array_match([4, 3, 2, 1]));

    assert_eq!(try_from_matched(*b"ABCD"), Ok(MyEnum::A));
    assert_eq!(try_from_matched(*b"EFGH"), Ok(MyEnum::B));
    assert_eq!(try_from_matched(*b"IJKL"), Ok(MyEnum::C));
    assert_eq!(try_from_matched(*b"MNOP"), Ok(MyEnum::D));
    assert_eq!(try_from_matched(*b"ZZZZ"), Err(()));
    assert_eq!(try_from_matched(*b"ABCE"), Err(()));
}
