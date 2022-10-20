#![feature(const_trait_impl, effects)]

// check-pass

#[const_trait]
trait Foo {
    type Assoc: ~const Foo;
    fn foo() {}
}

const fn foo<T: ~const Foo>() {
    <T as Foo>::Assoc::foo();
}

fn main() {}
