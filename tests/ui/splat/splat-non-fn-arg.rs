//! Test that using `#[splat]` on non-function-arguments is an error.

#![allow(incomplete_features)]
#![feature(splat)]

#[splat] //~ ERROR `#[splat]` attribute cannot be used on functions
fn tuple_args_bad((a, b): (u32, i8)) {}

#[splat] //~ ERROR `#[splat]` attribute cannot be used on traits
trait FooTraitBad {
    fn tuple_1(_: (u32,));

    fn tuple_4(self, _: (u32, i8, (), f32));
}

struct Foo;

#[splat] //~ ERROR `#[splat]` attribute cannot be used on inherent impl blocks
impl Foo {
    fn tuple_1_bad((a,): (u32,)) {}
}

impl Foo {
    #[splat] //~ ERROR `#[splat]` attribute cannot be used on inherent methods
    fn tuple_3_bad((a, b, c): (u32, i32, i8)) {}

    #[splat] //~ ERROR `#[splat]` attribute cannot be used on inherent methods
    fn tuple_2_bad(self, (a, b): (u32, i8)) -> u32 {
        a
    }
}

#[splat] //~ ERROR `#[splat]` attribute cannot be used on trait impl blocks
impl FooTraitBad for Foo {
    fn tuple_1(_: (u32,)) {}

    fn tuple_4(self, _: (u32, i8, (), f32)) {}
}

#[splat] //~ ERROR `#[splat]` attribute cannot be used on foreign modules
extern "C" {
    fn foo_2(_: (u32, i8));
}

extern "C" {
    #[splat] //~ ERROR `#[splat]` attribute cannot be used on foreign functions
    fn bar_2_bad(_: (u32, i8));
}

#[splat] //~ ERROR `#[splat]` attribute cannot be used on modules
mod foo_mod {}

#[splat] //~ ERROR `#[splat]` attribute cannot be used on use statements
use std::mem;

#[splat] //~ ERROR `#[splat]` attribute cannot be used on structs
struct FooStruct;

fn multisplat_bad(
    #[splat]
    #[splat] //~ WARN unused attribute
    (a, b): (u32, i8),
) {
}

fn main() {}
