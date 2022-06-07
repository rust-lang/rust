// check-pass
#![feature(const_trait_impl)]

#[const_trait]
trait Foo {
    fn foo();
}

const fn foo<T: ~const Foo>() {
    T::foo();
}

fn main() {}