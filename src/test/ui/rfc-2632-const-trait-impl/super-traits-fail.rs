#![feature(const_trait_impl)]

trait Foo {
    fn a(&self);
}
trait Bar: ~const Foo {}

struct S;
impl Foo for S {
    fn a(&self) {}
}

impl const Bar for S {}
//~^ ERROR the trait bound

fn main() {}
