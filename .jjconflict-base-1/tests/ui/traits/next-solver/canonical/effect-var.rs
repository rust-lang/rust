//@ compile-flags: -Znext-solver
//@ check-pass

#![feature(const_trait_impl)]

const trait Foo {
    fn foo();
}

trait Bar {}

const impl Foo for i32 {
    fn foo() {}
}

const impl<T> Foo for T
where
    T: Bar,
{
    fn foo() {}
}

fn main() {}
