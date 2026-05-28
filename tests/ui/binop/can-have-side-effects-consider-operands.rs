//@ check-pass

#![allow(unused)]

// Test if `Expr::can_have_side_effects` considers operands of binary operators.

fn drop_repeat_in_arm_body() {
    // Built-in lint `dropping_copy_types` relies on `Expr::can_have_side_effects`
    // (See rust-clippy#9482 and rust#113231)

    match () {
        () => drop(5 % 3), // No side effects
        //~^ WARNING calls to `std::mem::drop` with a value that implements `Copy` does nothing
    }
    match () {
        () => drop(5 % calls_are_considered_side_effects()), // Definitely has side effects
    }
}
fn calls_are_considered_side_effects() -> i32 {
    3
}

fn main() {}
