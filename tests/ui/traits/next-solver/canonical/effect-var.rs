//@ compile-flags: -Znext-solver
//@ check-pass

#![feature(const_trait_impl)]

#[const_trait]
trait Foo {
    (const) fn foo();
}

trait Bar {}

impl const Foo for i32 {
    (const) fn foo() {}
}

impl<T> const Foo for T where T: Bar {
    (const) fn foo() {}
}

fn main() {}
