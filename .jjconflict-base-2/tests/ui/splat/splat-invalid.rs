//! Test using `#[rustc_splat]` incorrectly, in ways not covered by other tests.

#![allow(incomplete_features)]
#![feature(splat)]

fn multisplat_fn_bad(#[rustc_splat] (_a, _b): (u32, i8), #[rustc_splat] (_c, _d): (u32, i8)) {}
//~^ ERROR multiple `#[rustc_splat]`s are not allowed in the same function argument list

fn multisplat_arg_bad(
    #[rustc_splat]
    #[rustc_splat]
    //~^ ERROR  multiple `rustc_splat` attributes
    (_a, _b): (u32, i8),
) {
}

fn multisplat_arg_fn_bad(
    #[rustc_splat]
    //~^ ERROR multiple `#[rustc_splat]`s are not allowed in the same function argument list
    #[rustc_splat]
    //~^ ERROR  multiple `rustc_splat` attributes
    (_a, _b): (u32, i8),
    #[rustc_splat] (_c, _d): (u32, i8),
) {
}

unsafe extern "C" fn splat_variadic(#[rustc_splat] (_a, _b): (u32, i8), varargs: ...) {}
//~^ ERROR `...` and `#[rustc_splat]` are not allowed in the same function argument list

unsafe extern "C" fn splat_variadic2(varargs: ..., #[rustc_splat] (_a, _b): (u32, i8)) {}
//~^ ERROR `...` and `#[rustc_splat]` are not allowed in the same function argument list
//~| ERROR `...` must be the last argument of a C-variadic function

extern "C" {
    fn splat_variadic3(#[rustc_splat] (_a, _b): (u32, i8), ...) {}
    //~^ ERROR incorrect function inside `extern` block
    //~| ERROR `...` and `#[rustc_splat]` are not allowed in the same function

    fn splat_variadic4(..., #[rustc_splat] (_a, _b): (u32, i8)) {}
    //~^ ERROR incorrect function inside `extern` block
    //~| ERROR `...` and `#[rustc_splat]` are not allowed in the same function
    //~| ERROR `...` must be the last argument of a C-variadic function

    // FIXME(splat): tuple layouts are unspecified. Should this error in addition to
    // the existing `improper_ctypes` lint?
    #[expect(improper_ctypes)]
    fn bar_2(#[rustc_splat] _: (u32, i8));
}

trait FooTrait {
    fn has_splat(#[rustc_splat] _: ());

    fn no_splat(_: (u32, f64));
}

struct Foo;

impl FooTrait for Foo {
    fn has_splat(_: ()) {} //~ ERROR method `has_splat` has an incompatible type for trait

    fn no_splat(#[rustc_splat] _: (u32, f64)) {} //~ ERROR method `no_splat` has an incompatible type for trait
}

#[rustfmt::skip]
fn main() {
    let multisplat_fn_bad_:
        fn(#[rustc_splat] (u32, i8), #[rustc_splat] (u32, i8)) = multisplat_fn_bad;
    //~^ ERROR multiple `#[rustc_splat]`s are not allowed in the same function argument list
    let multisplat_arg_bad_: fn(
        #[rustc_splat]
        #[rustc_splat]
        (u32, i8),
    ) = multisplat_arg_bad;
    let multisplat_arg_fn_bad_: fn(
        #[rustc_splat]
        //~^ ERROR multiple `#[rustc_splat]`s are not allowed in the same function argument list
        #[rustc_splat]
        (u32, i8),
        #[rustc_splat] (u32, i8),
    ) = multisplat_arg_fn_bad;

    let splat_variadic_: unsafe extern "C" fn(#[rustc_splat] (u32, i8), ...) = splat_variadic;
    //~^ ERROR `...` and `#[rustc_splat]` are not allowed in the same function argument list
    let splat_variadic2_: unsafe extern "C" fn(..., #[rustc_splat] (u32, i8)) = splat_variadic2;
    //~^ ERROR `...` must be the last argument of a C-variadic function
    //~| ERROR `...` and `#[rustc_splat]` are not allowed in the same function argument list
}
