// Issue #50124 - Test warning for unused operator expressions

//@ check-pass

#![warn(unused_must_use)]
#![feature(never_type)]

fn deref_never(x: &!) {
    // Don't lint for uninhabited types
    *x;
}

fn main() {
    let val = 1;
    let val_pointer = &val;

    // Comparison Operators
    val == 1; //~ WARNING unused comparison
    val < 1; //~ WARNING unused comparison
    val <= 1; //~ WARNING unused comparison
    val != 1; //~ WARNING unused comparison
    val >= 1; //~ WARNING unused comparison
    val > 1; //~ WARNING unused comparison

    // Arithmetic Operators
    val + 2; //~ WARNING unused arithmetic operation
    val - 2; //~ WARNING unused arithmetic operation
    val / 2; //~ WARNING unused arithmetic operation
    val * 2; //~ WARNING unused arithmetic operation
    val % 2; //~ WARNING unused arithmetic operation

    // Logical Operators
    true && true; //~ WARNING unused logical operation
    false || true; //~ WARNING unused logical operation

    // Bitwise Operators
    5 ^ val; //~ WARNING unused bitwise operation
    5 & val; //~ WARNING unused bitwise operation
    5 | val; //~ WARNING unused bitwise operation
    5 << val; //~ WARNING unused bitwise operation
    5 >> val; //~ WARNING unused bitwise operation

    // Unary Operators
    !val; //~ WARNING unused unary operation
    -val; //~ WARNING unused unary operation
    *val_pointer; //~ WARNING unused unary operation

    if false {
        deref_never(&panic!());
    }
}
