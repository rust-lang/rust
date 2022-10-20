#![feature(const_trait_impl)]
#![feature(effects)]

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
//~^ ERROR the trait bound

fn main() {}
