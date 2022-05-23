#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

fn main() {}

type OneTy<T> = impl Debug;

type OneLifetime<'a> = impl Debug;

type OneConst<const X: usize> = impl Debug;


// Not defining uses, because they doesn't define *all* possible generics.

fn concrete_ty() -> OneTy<u32> {
    5u32
    //~^ ERROR non-defining opaque type use in defining scope
}

fn concrete_lifetime() -> OneLifetime<'static> {
    6u32
    //~^ ERROR non-defining opaque type use in defining scope
}

fn concrete_const() -> OneConst<{ 123 }> {
    7u32
    //~^ ERROR non-defining opaque type use in defining scope
}
