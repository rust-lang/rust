#![warn(clippy::eq_op)]
#![allow(clippy::double_parens, clippy::identity_op, clippy::nonminimal_bool)]
#![allow(clippy::suspicious_xor_used_as_pow)]

fn main() {
    // simple values and comparisons
    let _ = 1 == 1;
    //~^ eq_op

    let _ = "no" == "no";
    //~^ eq_op

    // even though I agree that no means no ;-)
    let _ = false != false;
    //~^ eq_op

    let _ = 1.5 < 1.5;
    //~^ eq_op

    let _ = 1u64 >= 1u64;
    //~^ eq_op

    let x = f32::NAN;
    let _ = x != x;
    //~^ eq_op

    // casts, methods, parentheses
    let _ = (1u32 as u64) & (1u32 as u64);
    //~^ eq_op

    #[rustfmt::skip]
    {
        let _ = 1 ^ ((((((1))))));
        //~^ eq_op

    };

    // unary and binary operators
    let _ = (-(2) < -(2));
    //~^ eq_op

    let _ = ((1 + 1) & (1 + 1) == (1 + 1) & (1 + 1));
    //~^ eq_op
    //~| eq_op
    //~| eq_op

    let _ = (1 * 2) + (3 * 4) == 1 * 2 + 3 * 4;
    //~^ eq_op

    // various other things
    let _ = ([1] != [1]);
    //~^ eq_op

    let _ = ((1, 2) != (1, 2));
    //~^ eq_op

    let _ = vec![1, 2, 3] == vec![1, 2, 3]; //no error yet, as we don't match macros

    // const folding
    let _ = 1 + 1 == 2;
    //~^ eq_op

    let _ = 1 - 1 == 0;
    //~^ eq_op
    //~| eq_op

    let _ = 1 - 1;
    //~^ eq_op

    let _ = 1 / 1;
    //~^ eq_op

    let _ = true && true;
    //~^ eq_op

    let _ = true || true;
    //~^ eq_op

    let a: u32 = 0;
    let b: u32 = 0;

    let _ = a == b && b == a;
    //~^ eq_op

    let _ = a != b && b != a;
    //~^ eq_op

    let _ = a < b && b > a;
    //~^ eq_op

    let _ = a <= b && b >= a;
    //~^ eq_op

    let mut a = vec![1];
    let _ = a == a;
    //~^ eq_op

    let _ = 2 * a.len() == 2 * a.len(); // ok, functions
    let _ = a.pop() == a.pop(); // ok, functions

    check_ignore_macro();

    // named constants
    const A: u32 = 10;
    const B: u32 = 10;
    const C: u32 = A / B; // ok, different named constants
    const D: u32 = A / A;
    //~^ eq_op
}

macro_rules! check_if_named_foo {
    ($expression:expr) => {
        if stringify!($expression) == "foo" {
            println!("foo!");
        } else {
            println!("not foo.");
        }
    };
}

macro_rules! bool_macro {
    ($expression:expr) => {
        true
    };
}

fn check_ignore_macro() {
    check_if_named_foo!(foo);
    // checks if the lint ignores macros with `!` operator
    let _ = !bool_macro!(1) && !bool_macro!("");
}

struct Nested {
    inner: ((i32,), (i32,), (i32,)),
}

fn check_nested(n1: &Nested, n2: &Nested) -> bool {
    // `n2.inner.0.0` mistyped as `n1.inner.0.0`
    (n1.inner.0).0 == (n1.inner.0).0 && (n1.inner.1).0 == (n2.inner.1).0 && (n1.inner.2).0 == (n2.inner.2).0
    //~^ eq_op
}

#[test]
fn eq_op_shouldnt_trigger_in_tests() {
    let a = 1;
    let result = a + 1 == 1 + a;
    assert!(result);
}

#[test]
fn eq_op_macros_shouldnt_trigger_in_tests() {
    let a = 1;
    let b = 2;
    assert_eq!(a, a);
    assert_eq!(a + b, b + a);
}
