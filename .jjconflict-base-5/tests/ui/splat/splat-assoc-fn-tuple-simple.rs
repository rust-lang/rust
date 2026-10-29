//@ run-pass
//! Test using `#[rustc_splat]` on associated function tuple arguments (no receivers).

#![allow(incomplete_features)]
#![feature(splat)]

struct Foo;

impl Foo {
    fn tuple_1(#[rustc_splat] (_a,): (u32,)) {}

    fn tuple_3(#[rustc_splat] (_a, _b, _c): (u32, i32, i8)) {}
}

fn main() {
    Foo::tuple_1(1u32);
    Foo::tuple_3(1u32, 2i32, 3i8);
}
