//@ run-pass
//@ compile-flags: -O

// Make sure arithmetic unary/binary ops actually return the right result, even when overflowing.
// We have to put them in `const fn` and turn on optimizations to avoid overflow checks.

const fn add(x: i8, y: i8) -> i8 { x+y }
const fn sub(x: i8, y: i8) -> i8 { x-y }
const fn mul(x: i8, y: i8) -> i8 { x*y }
// div and rem are always checked, so we cannot test their result in case of overflow.
const fn neg(x: i8) -> i8 { -x }

fn main() {
    const ADD_OFLOW: i8 = add(100, 100);
    assert_eq!(ADD_OFLOW, -56);

    const SUB_OFLOW: i8 = sub(100, -100);
    assert_eq!(SUB_OFLOW, -56);

    const MUL_OFLOW: i8 = mul(-100, -2);
    assert_eq!(MUL_OFLOW, -56);

    const NEG_OFLOW: i8 = neg(-128);
    assert_eq!(NEG_OFLOW, -128);
}
