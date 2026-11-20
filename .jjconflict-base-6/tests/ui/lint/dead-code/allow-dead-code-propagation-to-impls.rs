//@ check-pass

#![deny(dead_code)]
#![deny(unfulfilled_lint_expectations)]

#[allow(dead_code)]
pub trait Tr {
    fn foo(&self);
}

#[expect(dead_code)]
struct Foo;

impl Tr for Foo {
    fn foo(&self) {
        bar();
    }
}

#[expect(dead_code)]
fn bar() {}

fn main() {}
