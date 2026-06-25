//@ run-pass
//! Test using `#[splat]` on associated function tuple arguments (no receivers).

#![allow(incomplete_features)]
#![feature(splat)]

struct Foo;

impl Foo {
    fn tuple_1(#[splat] (_a,): (u32,)) {}

    fn tuple_3(#[splat] (_a, _b, _c): (u32, i32, i8)) {}
}

fn main() {
    // FIXME(splat): should splatted functions be callable with tupled and un-tupled arguments?
    // Add a tupled test for each call if they are.
    //Foo::tuple_1((1u32,));

    Foo::tuple_1(1u32);
    Foo::tuple_3(1u32, 2i32, 3i8);
}
