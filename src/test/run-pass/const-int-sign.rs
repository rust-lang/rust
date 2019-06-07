#![feature(const_int_sign)]

const NEGATIVE_A: bool = (-10i32).is_negative();
const NEGATIVE_B: bool = 10i32.is_negative();
const POSITIVE_A: bool = (-10i32).is_positive();
const POSITIVE_B: bool = 10i32.is_positive();

const SIGNUM_POS: i32 = 10i32.signum();
const SIGNUM_NIL: i32 = 0i32.signum();
const SIGNUM_NEG: i32 = (-42i32).signum();

fn main() {
    assert!(NEGATIVE_A);
    assert!(!NEGATIVE_B);
    assert!(!POSITIVE_A);
    assert!(POSITIVE_B);

    assert_eq!(SIGNUM_POS, 1);
    assert_eq!(SIGNUM_NIL, 0);
    assert_eq!(SIGNUM_NEG, -1);
}
