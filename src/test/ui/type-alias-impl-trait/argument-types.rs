#![feature(type_alias_impl_trait)]
#![allow(dead_code)]
// check-pass
use std::fmt::Debug;

type Foo = impl Debug;

fn foo1(mut x: Foo) {
    x = 22_u32;
}

fn foo2(mut x: Foo) {
    // no constraint on x
}

fn foo3(x: Foo) {
    println!("{:?}", x);
}

fn foo_value() -> Foo {
    11_u32
}

fn main() {
    foo3(foo_value());
}
