// This test is to demonstrate that the lint is gated behind Edition and
// is triggered only for Edition 2021 and before.

//@ check-pass
//@ edition: 2024
//@ compile-flags: -Z unstable-options

#![deny(tail_expr_drop_order)]

struct LoudDropper;
impl Drop for LoudDropper {
    fn drop(&mut self) {
        // This destructor should be considered significant because it is a custom destructor
        // and we will assume that the destructor can generate side effects arbitrarily so that
        // a change in drop order is visible.
        println!("loud drop");
    }
}
impl LoudDropper {
    fn get(&self) -> i32 {
        0
    }
}

fn should_not_lint() -> i32 {
    let x = LoudDropper;
    x.get() + LoudDropper.get()
    // Lint should not action
}

fn main() {}
