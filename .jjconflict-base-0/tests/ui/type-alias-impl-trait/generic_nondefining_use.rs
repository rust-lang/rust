#![feature(type_alias_impl_trait)]

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

use std::fmt::Debug;

fn main() {}

type OneTy<T> = impl Debug;

type OneLifetime<'a> = impl Debug;

type OneConst<const X: usize> = impl Debug;

// Not defining uses, because they doesn't define *all* possible generics.

#[define_opaque(OneTy)]
fn concrete_ty() -> OneTy<u32> {
    //[current]~^ ERROR: expected generic type parameter, found `u32`
    //[next]~^^ ERROR: item does not constrain `OneTy::{opaque#0}`
    5u32
}

#[define_opaque(OneLifetime)]
fn concrete_lifetime() -> OneLifetime<'static> {
    //[next]~^ ERROR: non-defining use of `OneLifetime<'_>` in the defining scope
    6u32
    //[current]~^ ERROR: expected generic lifetime parameter, found `'static`

}

#[define_opaque(OneConst)]
fn concrete_const() -> OneConst<{ 123 }> {
    //[current]~^ ERROR: expected generic constant parameter, found `123`
    //[next]~^^ ERROR: item does not constrain `OneConst::{opaque#0}`
    7u32
}
