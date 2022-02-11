#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

// FIXME This should compile, but it currently doesn't

use std::fmt::Debug;

type Foo = impl Debug;
//~^ ERROR: could not find defining uses

fn foo1() -> u32 {
    let x: Foo = 22_u32;
    //~^ ERROR: mismatched types [E0308]
    x
    //~^ ERROR: mismatched types [E0308]
}

fn foo2() -> u32 {
    let x: Foo = 22_u32;
    //~^ ERROR: mismatched types [E0308]
    let y: Foo = x;
    same_type((x, y));
    y
    //~^ ERROR: mismatched types [E0308]
}

fn same_type<T>(x: (T, T)) {}

fn main() {}
