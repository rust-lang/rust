#![feature(type_alias_impl_trait)]
#![allow(dead_code)]
//@ check-pass

use std::fmt::Debug;

pub type Foo = impl Debug;

#[define_opaque(Foo)]
fn foo1(mut x: Foo) {
    x = 22_u32;
}

#[define_opaque(Foo)]
pub fn foo_value() -> Foo {
    11_u32
}

fn foo2(mut x: Foo) {
    // no constraint on x
}

fn foo3(x: Foo) {
    println!("{:?}", x);
}

fn main() {
    foo3(foo_value());
}
