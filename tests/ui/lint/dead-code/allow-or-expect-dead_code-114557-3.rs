//@ check-pass

// this test makes sure that the `unfulfilled_lint_expectations` lint
// is being emited for `foo` as foo is not dead code, it's pub

#![feature(lint_reasons)]
#![warn(dead_code)] // to override compiletest

#[expect(dead_code)]
//~^ WARN this lint expectation is unfulfilled
pub fn foo() {}

fn main() {}
