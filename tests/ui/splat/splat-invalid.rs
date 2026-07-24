//! Test using `#[arg_splat]` incorrectly, in ways not covered by other tests.

#![allow(incomplete_features)]
#![feature(arg_splat)]

fn multisplat_fn_bad(#[arg_splat] (_a, _b): (u32, i8), #[arg_splat] (_c, _d): (u32, i8)) {}
//~^ ERROR multiple `#[arg_splat]`s are not allowed in the same function argument list

fn multisplat_arg_bad(
    #[arg_splat]
    #[arg_splat]
    //~^ ERROR  multiple `arg_splat` attributes
    (_a, _b): (u32, i8),
) {
}

fn multisplat_arg_fn_bad(
    #[arg_splat]
    //~^ ERROR multiple `#[arg_splat]`s are not allowed in the same function argument list
    #[arg_splat]
    //~^ ERROR  multiple `arg_splat` attributes
    (_a, _b): (u32, i8),
    #[arg_splat] (_c, _d): (u32, i8),
) {
}

unsafe extern "C" fn splat_variadic(#[arg_splat] (_a, _b): (u32, i8), varargs: ...) {}
//~^ ERROR `...` and `#[arg_splat]` are not allowed in the same function argument list

unsafe extern "C" fn splat_variadic2(varargs: ..., #[arg_splat] (_a, _b): (u32, i8)) {}
//~^ ERROR `...` and `#[arg_splat]` are not allowed in the same function argument list
//~| ERROR `...` must be the last argument of a C-variadic function

extern "C" {
    fn splat_variadic3(#[arg_splat] (_a, _b): (u32, i8), ...) {}
    //~^ ERROR incorrect function inside `extern` block
    //~| ERROR `...` and `#[arg_splat]` are not allowed in the same function

    fn splat_variadic4(..., #[arg_splat] (_a, _b): (u32, i8)) {}
    //~^ ERROR incorrect function inside `extern` block
    //~| ERROR `...` and `#[arg_splat]` are not allowed in the same function
    //~| ERROR `...` must be the last argument of a C-variadic function

    // FIXME(splat): tuple layouts are unspecified. Should this error in addition to
    // the existing `improper_ctypes` lint?
    #[expect(improper_ctypes)]
    fn bar_2(#[arg_splat] _: (u32, i8));
}

trait FooTrait {
    fn has_splat(#[arg_splat] _: ());

    fn no_splat(_: (u32, f64));
}

struct Foo;

impl FooTrait for Foo {
    fn has_splat(_: ()) {} //~ ERROR method `has_splat` has an incompatible type for trait

    fn no_splat(#[arg_splat] _: (u32, f64)) {} //~ ERROR method `no_splat` has an incompatible type for trait
}

fn main() {}
