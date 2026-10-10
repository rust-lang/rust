//@ check-pass

#![deny(unfulfilled_lint_expectations)]
#![warn(dead_code)]

#[expect(dead_code)]
fn root() {
    middle();
}

fn middle() {
    leaf();
}

#[expect(dead_code)]
fn leaf() {}

fn main() {}
