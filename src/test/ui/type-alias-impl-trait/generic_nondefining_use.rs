#![feature(type_alias_impl_trait, const_generics)]
#![allow(incomplete_features)]

use std::fmt::Debug;

fn main() {}

type OneTy<T> = impl Debug;
type OneLifetime<'a> = impl Debug;
type OneConst<const X: usize> = impl Debug;

// Not defining uses, because they doesn't define *all* possible generics.

fn concrete_ty() -> OneTy<u32> {
//~^ ERROR non-defining opaque type use in defining scope
    5u32
}

fn concrete_lifetime() -> OneLifetime<'static> {
//~^ ERROR non-defining opaque type use in defining scope
    6u32
}

fn concrete_const() -> OneConst<{123}> {
//~^ ERROR non-defining opaque type use in defining scope
    7u32
}
