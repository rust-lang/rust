//@ check-pass

// Test that the def collector makes `AnonConst`s not `InlineConst`s even
// when the const block is obscured via macros.

#![feature(min_generic_const_args, adt_const_params)]
#![expect(incomplete_features)]

macro_rules! const_block {
    ($e:expr) => { const {
        $e
    } }
}

macro_rules! foo_expr {
    ($e:expr) => { Foo {
        field: $e,
    } }
}

use std::marker::ConstParamTy;

#[derive(PartialEq, Eq, ConstParamTy)]
struct Foo { field: u32 }

fn foo<const N: Foo>() {}

fn main() {
    foo::<{ Foo { field: const_block!{ 1 + 1 }} }>();
    foo::<{ foo_expr! { const_block! { 1 + 1 }} }>();
}
