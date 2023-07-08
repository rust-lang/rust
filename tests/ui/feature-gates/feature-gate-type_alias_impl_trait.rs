// check-pass
#![feature(type_alias_impl_trait)]
use std::fmt::Debug;

type Foo = impl Debug;

struct Bar(Foo);
fn define() -> Bar {
    Bar(42)
}

type Foo2 = impl Debug;

fn define2(_: Foo2) {
    let x = || -> Foo2 { 42 };
}

type Foo3 = impl Debug;

fn define3(x: Foo3) {
    let y: i32 = x;
}
fn define3_1(_: Foo3) {
    define3(42)
}

type Foo4 = impl Debug;

fn define4(_: Foo4) {
    let y: Foo4 = 42;
}

fn main() {}
