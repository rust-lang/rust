//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

use std::fmt::Debug;

type Foo = impl Debug;

#[define_opaque(Foo)]
fn foo1() -> (u32, Foo) {
    let x: Foo = 22_u32;
    (x, todo!())
}

#[define_opaque(Foo)]
fn foo2() -> (u32, Foo) {
    let x: Foo = 22_u32;
    let y: Foo = x;
    same_type((x, y)); //[current]~ ERROR use of moved value
    (y, todo!()) //[current]~ ERROR use of moved value
}

fn same_type<T>(x: (T, T)) {}

fn main() {}
