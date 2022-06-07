// check-pass
#![feature(const_trait_impl)]

#[const_trait]
trait Foo {
    fn foo();
}

pub struct W<T>(T);

impl<T: ~const Foo> const Foo for W<T> {
    fn foo() {
        <T as Foo>::foo();
    }
}

fn main() {}
