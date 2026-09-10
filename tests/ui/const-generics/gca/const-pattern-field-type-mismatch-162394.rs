//@ compile-flags: -Znext-solver=globally

#![allow(incomplete_features)]
#![feature(macroless_generic_const_args)]
#![feature(generic_const_args, min_generic_const_args)]
#![feature(min_adt_const_params)]

use std::marker::ConstParamTy;

#[derive(PartialEq, Eq, ConstParamTy)]
pub enum Foo {
    FooA(()),
}

impl Foo {
    const A2: Foo = Self::FooA("foo"); //~ ERROR the constant `"foo"` is not of type `()`
}

fn main() {
    let foo = Foo::FooA(());
    match foo {
        Foo::A2 => {}
        _ => {}
    }
}
