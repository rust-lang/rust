//@ check-pass

#![allow(unused)]

// Test if `Expr::can_have_side_effects` considers element of repeat expressions.

fn drop_repeat_in_arm_body() {
    // Built-in lint `dropping_copy_types` relies on `Expr::can_have_side_effects`
    // (See rust-clippy#9482 and rust#113231)

    match () {
        () => drop([0; 1]), // No side effects
        //~^ WARNING calls to `std::mem::drop` with a value that implements `Copy` does nothing
    }
    match () {
        () => drop([return; 1]), // Definitely has side effects
    }
}

fn main() {}
