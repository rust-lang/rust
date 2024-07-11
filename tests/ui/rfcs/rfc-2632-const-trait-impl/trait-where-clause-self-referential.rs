//@ check-pass
//@ compile-flags: -Znext-solver
#![allow(incomplete_features)]
#![feature(const_trait_impl, effects)]

#[const_trait]
trait Foo {
    fn bar() where Self: ~const Foo;
}

struct S;

impl Foo for S {
    fn bar() {}
}

fn baz<T: Foo>() {
    T::bar();
}

const fn qux<T: ~const Foo>() {
    T::bar();
}

fn main() {}
