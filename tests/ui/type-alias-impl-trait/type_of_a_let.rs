#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

use std::fmt::Debug;

type Foo = impl Debug;

#[defines(Foo)]
fn foo1() -> u32 {
    let x: Foo = 22_u32;
    x
}

#[defines(Foo)]
fn foo2() -> u32 {
    let x: Foo = 22_u32;
    let y: Foo = x;
    same_type((x, y)); //~ ERROR use of moved value
    y //~ ERROR use of moved value
}

fn same_type<T>(x: (T, T)) {}

fn main() {}
