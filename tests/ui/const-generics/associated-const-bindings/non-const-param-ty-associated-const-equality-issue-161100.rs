//@ compile-flags: -Znext-solver=globally

#![feature(generic_const_args)]
#![feature(min_generic_const_args)]
#![expect(incomplete_features)]
#![allow(dead_code)]

enum Foo {
    Unit,
    Function(fn()),
}

trait Trait {
    const X: Foo;
}

fn unit(_: impl Trait<X = { Foo::Unit }>) {}
//~^ ERROR `Foo` must implement `ConstParamTy`

fn main() {}
