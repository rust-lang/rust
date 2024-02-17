//@ check-pass
//@ known-bug: #110395

#![feature(const_trait_impl)]

#[const_trait]
trait Foo {
    fn a(&self);
}
#[const_trait]
trait Bar: ~const Foo {}

struct S;
impl Foo for S {
    fn a(&self) {}
}

impl const Bar for S {}
//FIXME ~^ ERROR the trait bound

fn main() {}
