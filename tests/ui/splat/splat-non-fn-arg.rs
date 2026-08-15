//! Test that using `#[rustc_splat]` on non-function-arguments is an error.

#![allow(incomplete_features)]
#![feature(splat)]

#[rustc_splat] //~ ERROR the `rustc_splat` attribute cannot be used on functions
fn tuple_args_bad((a, b): (u32, i8)) {}

#[rustc_splat] //~ ERROR the `rustc_splat` attribute cannot be used on traits
trait FooTraitBad {
    fn tuple_1(_: (u32,));

    fn tuple_4(self, _: (u32, i8, (), f32));
}

struct Foo;

#[rustc_splat] //~ ERROR the `rustc_splat` attribute cannot be used on inherent impl blocks
impl Foo {
    fn tuple_1_bad((a,): (u32,)) {}
}

impl Foo {
    #[rustc_splat] //~ ERROR the `rustc_splat` attribute cannot be used on inherent methods
    fn tuple_3_bad((a, b, c): (u32, i32, i8)) {}

    #[rustc_splat] //~ ERROR the `rustc_splat` attribute cannot be used on inherent methods
    fn tuple_2_bad(self, (a, b): (u32, i8)) -> u32 {
        a
    }
}

#[rustc_splat] //~ ERROR the `rustc_splat` attribute cannot be used on trait impl blocks
impl FooTraitBad for Foo {
    fn tuple_1(_: (u32,)) {}

    fn tuple_4(self, _: (u32, i8, (), f32)) {}
}

#[rustc_splat] //~ ERROR the `rustc_splat` attribute cannot be used on foreign modules
extern "C" {
    fn foo_2(_: (u32, i8));
}

extern "C" {
    #[rustc_splat] //~ ERROR the `rustc_splat` attribute cannot be used on foreign functions
    fn bar_2_bad(_: (u32, i8));
}

#[rustc_splat] //~ ERROR the `rustc_splat` attribute cannot be used on modules
mod foo_mod {}

#[rustc_splat] //~ ERROR the `rustc_splat` attribute cannot be used on use statements
use std::mem;

#[rustc_splat] //~ ERROR the `rustc_splat` attribute cannot be used on structs
struct FooStruct;

fn main() {}
