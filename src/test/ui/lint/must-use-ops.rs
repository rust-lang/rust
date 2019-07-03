// Issue #50124 - Test warning for unused operator expressions

// build-pass (FIXME(62277): could be check-pass?)

#![warn(unused_must_use)]

fn main() {
    let val = 1;
    let val_pointer = &val;

// Comparison Operators
    val == 1;
    val < 1;
    val <= 1;
    val != 1;
    val >= 1;
    val > 1;

// Arithmetic Operators
    val + 2;
    val - 2;
    val / 2;
    val * 2;
    val % 2;

// Logical Operators
    true && true;
    false || true;

// Bitwise Operators
    5 ^ val;
    5 & val;
    5 | val;
    5 << val;
    5 >> val;

// Unary Operators
    !val;
    -val;
    *val_pointer;
}
