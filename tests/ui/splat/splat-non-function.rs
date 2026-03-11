#![allow(incomplete_features)]
#![feature(splat)]

fn tuple_args(#[splat] (a, b): (u32, i8)) {}

#[splat] //~ ERROR `#[splat]` attribute cannot be used on functions
fn tuple_args_bad((a, b): (u32, i8)) {}

#[splat] //~ ERROR `#[splat]` attribute cannot be used on traits
trait FooTraitBad {
    fn tuple_1(_: (u32,));

    fn tuple_4(self, _: (u32, i8, (), f32));
}

trait FooTrait {
    fn tuple_1(#[splat] _: (u32,));

    fn tuple_4(#[splat] self, _: (u32, i8, (), f32));
}

struct Foo;

#[splat] //~ ERROR `#[splat]` attribute cannot be used on inherent impl blocks
impl Foo {
    fn tuple_1_bad((a,): (u32,)) {}

    fn tuple_4_bad(self, (a, b, c, d): (u32, i8, (), f32)) -> u32 {
        a
    }
}

impl Foo {
    #[splat] //~ ERROR `#[splat]` attribute cannot be used on inherent methods
    fn tuple_3_bad((a, b, c): (u32, i32, i8)) {}

    #[splat] //~ ERROR `#[splat]` attribute cannot be used on inherent methods
    fn tuple_2_bad(self, (a, b): (u32, i8)) -> u32 {
        a
    }

    fn tuple_1(#[splat] (a,): (u32,)) {}

    // FIXME(splat): this should error except when `self` (or any splatted arg) is a tuple.
    // Tuple structs should also error until we have a specific use case for them, and so should
    // multiple splats in a fn.
    fn tuple_2_self(#[splat] self, (a, b): (u32, i8)) -> u32 {
        a
    }

    fn tuple_3(#[splat] (a, b, c): (u32, i32, i8)) {}

    fn tuple_2(self, #[splat] (a, b): (u32, i8)) -> u32 {
        a
    }

    fn tuple_4(self, #[splat] (a, b, c, d): (u32, i8, (), f32)) -> u32 {
        a
    }
}

impl FooTrait for Foo {
    // FIXME(splat): should conflicting splat attributes be allowed on traits and impls?
    fn tuple_1(_: (u32,)) {}

    fn tuple_4(#[splat] self, _: (u32, i8, (), f32)) {}
}

#[splat] //~ ERROR `#[splat]` attribute cannot be used on foreign modules
extern "C" {
    fn foo_2(_: (u32, i8));
}

extern "C" {
    fn bar_2(#[splat] _: (u32, i8));

    #[splat] //~ ERROR `#[splat]` attribute cannot be used on foreign functions
    fn bar_2_bad(_: (u32, i8));
}

#[splat] //~ ERROR `#[splat]` attribute cannot be used on modules
mod foo_mod {}

#[splat] //~ ERROR `#[splat]` attribute cannot be used on use statements
use std::mem;

#[splat] //~ ERROR `#[splat]` attribute cannot be used on structs
struct FooStruct;

fn main() {}
