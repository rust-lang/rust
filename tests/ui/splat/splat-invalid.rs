//! Test using `#[splat]` incorrectly, in ways not covered by other tests.

#![allow(incomplete_features)]
#![feature(splat)]
#![feature(c_variadic)]

fn multisplat_bad(#[splat] (_a, _b): (u32, i8), #[splat] (_c, _d): (u32, i8)) {}
//~^ ERROR multiple `#[splat]`s are not allowed in the same function

unsafe extern "C" fn splat_variadic(#[splat] (_a, _b): (u32, i8), varargs: ...) {}
//~^ ERROR `...` and `#[splat]` are not allowed in the same function

unsafe extern "C" fn splat_variadic2(varargs: ..., #[splat] (_a, _b): (u32, i8)) {}
//~^ ERROR `...` and `#[splat]` are not allowed in the same function
//~| ERROR `...` must be the last argument of a C-variadic function

extern "C" {
    fn splat_variadic3(#[splat] (_a, _b): (u32, i8), ...) {}
    //~^ ERROR incorrect function inside `extern` block
    //~| ERROR `...` and `#[splat]` are not allowed in the same function

    fn splat_variadic4(..., #[splat] (_a, _b): (u32, i8)) {}
    //~^ ERROR incorrect function inside `extern` block
    //~| ERROR `...` and `#[splat]` are not allowed in the same function
    //~| ERROR `...` must be the last argument of a C-variadic function

    // FIXME(splat): tuple layouts are unspecified. Should this error in addition to
    // the existing `improper_ctypes` lint?
    #[expect(improper_ctypes)]
    fn bar_2(#[splat] _: (u32, i8));
}

trait FooTrait {
    fn has_splat(#[splat] _: ());

    fn no_splat(_: (u32, f64));
}

struct Foo;

impl FooTrait for Foo {
    fn has_splat(_: ()) {} //~ ERROR method `has_splat` has an incompatible type for trait

    fn no_splat(#[splat] _: (u32, f64)) {} //~ ERROR method `no_splat` has an incompatible type for trait
}

fn main() {}
