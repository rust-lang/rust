#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

// FIXME This should be under a feature flag

use std::fmt::Debug;

fn foo1() -> u32 {
    let x: impl Debug = 22_u32;
    //~^ ERROR: `impl Trait` not allowed outside of function and method return types [E0562]
    x // ERROR: we only know x: Debug, we don't know x = u32
}

fn foo2() -> u32 {
    let x: impl Debug = 22_u32;
    //~^ ERROR: `impl Trait` not allowed outside of function and method return types [E0562]
    let y: impl Debug = x;
    //~^ ERROR: `impl Trait` not allowed outside of function and method return types [E0562]
    same_type((x, y)); // ERROR
    x
}

fn same_type<T>(x: (T, T)) {}

fn main() {}
