//@ check-pass

#![deny(dead_code, unfulfilled_lint_expectations, reason = "example")]
#![expect(dead_code, reason = "example")]

struct Foo;
impl Foo {}

fn main() {}
