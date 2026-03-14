#![allow(incomplete_features)]
#![feature(splat)]
use std::splat::splat;

#[splat]
fn tuple_args((a, b): (u32, i8)) {}

#[splat]
trait FooTrait {
    fn tuple_1(_: (u32,));

    fn tuple_4(self, _: (u32, i8, (), f32));
}

struct GoodFoo;

#[splat]
impl GoodFoo {
    fn tuple_1((a,): (u32,)) {}

    fn tuple_4(self, (a, b, c, d): (u32, i8, (), f32)) -> u32 {
        a
    }
}

impl GoodFoo {
    #[splat]
    fn tuple_3((a, b, c): (u32, i32, i8)) {}

    #[splat]
    fn tuple_2(self, (a, b): (u32, i8)) -> u32 {
        a
    }
}

#[splat]
extern "C" {
    fn foo_2(_: (u32, i8));
}

// FIXME(splat): this might not be possible if we need the extern "C" for the trampoline
extern "C" {
    #[splat]
    fn bar_2(_: (u32, i8));
}

#[splat]
mod foo_mod {} //~ ERROR `#[splat]` attribute is only allowed on functions

#[splat]
extern crate foo_crate; //~ ERROR `#[splat]` attribute is only allowed on functions

#[splat]
use std::mem; //~ ERROR `#[splat]` attribute is only allowed on functions

#[splat]
type FooTy = u32; //~ ERROR `#[splat]` attribute is only allowed on functions

#[splat]
struct FooStruct; //~ ERROR `#[splat]` attribute is only allowed on functions

#[splat]
const FOO: () = (); //~ ERROR `#[splat]` attribute is only allowed on functions

fn main() {}
