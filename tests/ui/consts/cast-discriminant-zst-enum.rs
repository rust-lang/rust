//@ run-pass
// Test a ZST enum whose dicriminant is ~0i128. This caused an ICE when casting to an i32.
use std::hint::black_box;

#[derive(Copy, Clone)]
enum Nums {
    NegOne = -1,
}

const NEG_ONE_I8: i8 = Nums::NegOne as i8;
const NEG_ONE_I16: i16 = Nums::NegOne as i16;
const NEG_ONE_I32: i32 = Nums::NegOne as i32;
const NEG_ONE_I64: i64 = Nums::NegOne as i64;
const NEG_ONE_I128: i128 = Nums::NegOne as i128;

fn test_as_arg(n: Nums) {
    assert_eq!(-1i8, n as i8);
    assert_eq!(-1i16, n as i16);
    assert_eq!(-1i32, n as i32);
    assert_eq!(-1i64, n as i64);
    assert_eq!(-1i128, n as i128);
}

fn main() {
    let kind = Nums::NegOne;
    assert_eq!(-1i8, kind as i8);
    assert_eq!(-1i16, kind as i16);
    assert_eq!(-1i32, kind as i32);
    assert_eq!(-1i64, kind as i64);
    assert_eq!(-1i128, kind as i128);

    assert_eq!(-1i8, black_box(kind) as i8);
    assert_eq!(-1i16, black_box(kind) as i16);
    assert_eq!(-1i32, black_box(kind) as i32);
    assert_eq!(-1i64, black_box(kind) as i64);
    assert_eq!(-1i128, black_box(kind) as i128);

    test_as_arg(Nums::NegOne);

    assert_eq!(-1i8, NEG_ONE_I8);
    assert_eq!(-1i16, NEG_ONE_I16);
    assert_eq!(-1i32, NEG_ONE_I32);
    assert_eq!(-1i64, NEG_ONE_I64);
    assert_eq!(-1i128, NEG_ONE_I128);
}
