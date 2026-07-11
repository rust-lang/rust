//! Verify that matching against array/slice patterns expanded from constants
//! produces correct results at runtime, complementing the MIR test in
//! `tests/mir-opt/building/match/aggregate_array_eq.rs` which checks that
//! a single aggregate `PartialEq::eq` call is emitted.
//!
//! Also verify that the variants which fall back to element-by-element
//! comparison (hand-written patterns and const contexts) produce the same
//! results.
//@ run-pass

const EXPECTED: [u8; 4] = [1, 2, 3, 4];

fn array_match(x: [u8; 4]) -> bool {
    matches!(x, EXPECTED)
}

fn handwritten_array_match(x: [u8; 4]) -> bool {
    matches!(x, [1, 2, 3, 4])
}

fn slice_match(x: &[u8]) -> bool {
    matches!(x, b"ABCD")
}

const NESTED: [[u8; 4]; 4] = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]];

fn nested_array_match(x: [[u8; 4]; 4]) -> bool {
    matches!(x, NESTED)
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
    matches!(x, EXPECTED)
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

    assert!(handwritten_array_match([1, 2, 3, 4]));
    assert!(!handwritten_array_match([1, 2, 3, 5]));

    assert!(slice_match(b"ABCD"));
    assert!(!slice_match(b"ABCE"));
    assert!(!slice_match(b"ABC"));
    assert!(!slice_match(b"ABCDE"));
    assert!(!slice_match(b""));

    assert!(nested_array_match(NESTED));
    assert!(!nested_array_match([[0; 4]; 4]));
    assert!(!nested_array_match([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]));

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
