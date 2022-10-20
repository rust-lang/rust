#![feature(const_trait_impl, effects)]

#[const_trait]
trait Tr {}
impl Tr for () {}

const fn foo<T>() where T: ~const Tr {}

#[const_trait]
pub trait Foo {
    fn foo() {
        foo::<()>();
        //~^ ERROR cannot call
    }
}

fn main() {}
