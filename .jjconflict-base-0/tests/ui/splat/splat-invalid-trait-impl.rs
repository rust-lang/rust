//! Test that `#[rustc_splat]` trait impls with mismatched tuple element types are rejected.
#![allow(incomplete_features)]
#![feature(splat)]

trait FooTrait {
    fn method(#[rustc_splat] _: (u32, i8));
}

struct Foo;
struct Foo1;
struct Foo2;

impl FooTrait for Foo {
    fn method(#[rustc_splat] _: (u32, f32)) {}
    //~^ ERROR method `method` has an incompatible type for trait
}

impl FooTrait for Foo1 {
    fn method(#[rustc_splat] _: (f32, i8)) {}
    //~^ ERROR method `method` has an incompatible type for trait
}

impl FooTrait for Foo2 {
    fn method(#[rustc_splat] _: (f32, f64)) {}
    //~^ ERROR method `method` has an incompatible type for trait
}
fn main() {}
