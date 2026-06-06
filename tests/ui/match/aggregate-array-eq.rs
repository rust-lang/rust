//! Verify that matching against a constant array pattern produces correct
//! results at runtime, complementing the MIR test in
//! `tests/mir-opt/building/match/aggregate_array_eq.rs` which checks that
//! a single aggregate `PartialEq::eq` call is emitted.
//!
//! Also verify that const-context variants (which fall back to
//! element-by-element comparison) produce the same results.
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

// Const fn variants use element-by-element comparison because
// `PartialEq::eq` is not available in const contexts.
const fn const_array_match(x: [u8; 4]) -> bool {
    matches!(x, [1, 2, 3, 4])
}

const fn const_try_from_matched(value: [u8; 4]) -> Result<MyEnum, ()> {
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

    // Const fn variants called at runtime.
    assert!(const_array_match([1, 2, 3, 4]));
    assert!(!const_array_match([1, 2, 3, 5]));
    assert!(!const_array_match([0, 0, 0, 0]));
    assert!(!const_array_match([4, 3, 2, 1]));

    assert_eq!(const_try_from_matched(*b"ABCD"), Ok(MyEnum::A));
    assert_eq!(const_try_from_matched(*b"EFGH"), Ok(MyEnum::B));
    assert_eq!(const_try_from_matched(*b"IJKL"), Ok(MyEnum::C));
    assert_eq!(const_try_from_matched(*b"MNOP"), Ok(MyEnum::D));
    assert_eq!(const_try_from_matched(*b"ZZZZ"), Err(()));
    assert_eq!(const_try_from_matched(*b"ABCE"), Err(()));

    // Const fn variants evaluated at compile time.
    const MATCH_TRUE: bool = const_array_match([1, 2, 3, 4]);
    const MATCH_FALSE: bool = const_array_match([1, 2, 3, 5]);
    assert!(MATCH_TRUE);
    assert!(!MATCH_FALSE);

    const FROM_ABCD: Result<MyEnum, ()> = const_try_from_matched(*b"ABCD");
    const FROM_MNOP: Result<MyEnum, ()> = const_try_from_matched(*b"MNOP");
    const FROM_ZZZZ: Result<MyEnum, ()> = const_try_from_matched(*b"ZZZZ");
    assert_eq!(FROM_ABCD, Ok(MyEnum::A));
    assert_eq!(FROM_MNOP, Ok(MyEnum::D));
    assert_eq!(FROM_ZZZZ, Err(()));
}
