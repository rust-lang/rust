//@ check-pass

// this test checks that the `dead_code` lint is *NOT* being emited
// for `foo` as `foo` is being used by `main`, and so the `#[expect]`
// is unfulfilled
//
// it also checks that the `dead_code` lint is also *NOT* emited
// for `bar` as it's suppresed by the `#[expect]` on `bar`

#![feature(lint_reasons)]
#![warn(dead_code)] // to override compiletest

fn bar() {}

#[expect(dead_code)]
//~^ WARN this lint expectation is unfulfilled
fn foo() { bar() }

fn main() { foo() }
