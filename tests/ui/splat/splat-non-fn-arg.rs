//! Test that using `#[arg_splat]` on non-function-arguments is an error.

#![allow(incomplete_features)]
#![feature(arg_splat)]

#[arg_splat] //~ ERROR `#[arg_splat]` attribute cannot be used on functions
fn tuple_args_bad((a, b): (u32, i8)) {}

#[arg_splat] //~ ERROR `#[arg_splat]` attribute cannot be used on traits
trait FooTraitBad {
    fn tuple_1(_: (u32,));

    fn tuple_4(self, _: (u32, i8, (), f32));
}

struct Foo;

#[arg_splat] //~ ERROR `#[arg_splat]` attribute cannot be used on inherent impl blocks
impl Foo {
    fn tuple_1_bad((a,): (u32,)) {}
}

impl Foo {
    #[arg_splat] //~ ERROR `#[arg_splat]` attribute cannot be used on inherent methods
    fn tuple_3_bad((a, b, c): (u32, i32, i8)) {}

    #[arg_splat] //~ ERROR `#[arg_splat]` attribute cannot be used on inherent methods
    fn tuple_2_bad(self, (a, b): (u32, i8)) -> u32 {
        a
    }
}

#[arg_splat] //~ ERROR `#[arg_splat]` attribute cannot be used on trait impl blocks
impl FooTraitBad for Foo {
    fn tuple_1(_: (u32,)) {}

    fn tuple_4(self, _: (u32, i8, (), f32)) {}
}

#[arg_splat] //~ ERROR `#[arg_splat]` attribute cannot be used on foreign modules
extern "C" {
    fn foo_2(_: (u32, i8));
}

extern "C" {
    #[arg_splat] //~ ERROR `#[arg_splat]` attribute cannot be used on foreign functions
    fn bar_2_bad(_: (u32, i8));
}

#[arg_splat] //~ ERROR `#[arg_splat]` attribute cannot be used on modules
mod foo_mod {}

#[arg_splat] //~ ERROR `#[arg_splat]` attribute cannot be used on use statements
use std::mem;

#[arg_splat] //~ ERROR `#[arg_splat]` attribute cannot be used on structs
struct FooStruct;

fn main() {}
