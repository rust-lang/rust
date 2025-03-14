//@ check-pass
#![feature(type_alias_impl_trait)]
use std::fmt::Debug;

type Foo = impl Debug;

struct Bar(Foo);
#[define_opaque(Foo)]
fn define() -> Bar {
    Bar(42)
}

type Foo2 = impl Debug;

#[define_opaque(Foo2)]
fn define2() {
    let x = || -> Foo2 { 42 };
}

type Foo3 = impl Debug;

#[define_opaque(Foo3)]
fn define3(x: Foo3) {
    let y: i32 = x;
}
#[define_opaque(Foo3)]
fn define3_1() {
    define3(42)
}

type Foo4 = impl Debug;

#[define_opaque(Foo4)]
fn define4(_: Foo4) {
    let y: Foo4 = 42;
}

fn main() {}
