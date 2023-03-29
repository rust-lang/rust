#![feature(type_alias_impl_trait)]
#![allow(dead_code)]
// check-pass
use std::fmt::Debug;

type Foo = impl Debug;

#[defines(Foo)]
fn foo1(mut x: Foo) {
    x = 22_u32;
}

#[defines(Foo)]
fn foo2(mut x: Foo) {
    // no constraint on x
}

#[defines(Foo)]
fn foo3(x: Foo) {
    println!("{:?}", x);
}

#[defines(Foo)]
fn foo_value() -> Foo {
    11_u32
}

#[defines(Foo)]
fn main() {
    foo3(foo_value());
}
