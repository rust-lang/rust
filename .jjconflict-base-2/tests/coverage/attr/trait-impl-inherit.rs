#![feature(coverage_attribute)]
// Checks that `#[coverage(..)]` in a trait method is not inherited in an
// implementation.
//@ edition: 2021
//@ reference: attributes.coverage.trait-impl-inherit

trait T {
    #[coverage(off)]
    fn f(&self) {
        println!("default");
    }
}

struct S;

impl T for S {
    fn f(&self) {
        println!("impl S");
    }
}

#[coverage(off)]
fn main() {
    S.f();
}
