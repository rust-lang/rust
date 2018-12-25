#![allow(const_err)]

// error-pattern: attempt to divide by zero

fn main() {
    let x = &(1 / (1 - 1));
}
