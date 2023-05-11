// check-pass
#![feature(const_trait_impl)]

#[const_trait]
trait Foo {
    type Assoc: ~const Foo;
    fn foo() {}
}

const fn foo<T: ~const Foo>() {
    <T as Foo>::Assoc::foo();
}

fn main() {}
