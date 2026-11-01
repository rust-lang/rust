// This test ensures that `tail_expr_drop_order` does not activate in case Edition 2024 is used
// because this is a migration lint.
// Only `cargo fix --edition 2024` shall activate this lint.

//@ check-pass
//@ edition: 2024

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
