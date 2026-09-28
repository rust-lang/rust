//@ compile-flags: -Znext-solver=globally

#![feature(
    gca_adts,
    gca_const_items,
    gca_macroless_args,
    gca_macroless_items,
    gca_min_const_items,
    min_adt_const_params
)]

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
