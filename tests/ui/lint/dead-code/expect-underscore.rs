#![deny(unfulfilled_lint_expectations)]

#[expect(dead_code)] //~ ERROR: this lint expectation is unfulfilled
fn _unused() {}

fn main() {}
