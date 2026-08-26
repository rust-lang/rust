//@ compile-flags: -Znext-solver=globally

#![feature(gca_const_items)]
#![feature(gca_min_const_items)]

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
