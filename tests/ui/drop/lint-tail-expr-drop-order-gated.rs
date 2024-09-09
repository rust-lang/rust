// This test ensures that `tail_expr_drop_order` does not activate in case Edition 2024 is not used
// or the feature gate `shorter_tail_lifetimes` is disabled.

//@ revisions: neither no_feature_gate edition_less_than_2024
//@ check-pass
//@ [neither] edition: 2021
//@ [no_feature_gate] compile-flags: -Z unstable-options
//@ [no_feature_gate] edition: 2024
//@ [edition_less_than_2024] edition: 2021

#![deny(tail_expr_drop_order)]
#![cfg_attr(edition_less_than_2024, feature(shorter_tail_lifetimes))]

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
