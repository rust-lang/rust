//@ compile-flags: -Znext-solver

#![feature(const_trait_impl)]

const trait Foo {
    fn a(&self);
}
const trait Bar: [const] Foo {}

struct S;
impl Foo for S {
    fn a(&self) {}
}

impl const Bar for S {}
//~^ ERROR the trait bound

fn main() {}
