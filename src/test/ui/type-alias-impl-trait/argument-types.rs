#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

use std::fmt::Debug;

type Foo = impl Debug;

// FIXME: This should compile, but it currently doesn't
fn foo1(mut x: Foo) {
    x = 22_u32;
    //~^ ERROR: mismatched types [E0308]
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
