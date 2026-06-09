//@ check-fail
//@ compile-flags: --crate-type=lib
#![feature(c_variadic)]
#![feature(rustc_attrs)]

#[rustc_no_mir_inline]
#[rustc_force_inline]
//~^ ERROR `rustc_attr` is incompatible with `#[rustc_force_inline]`
pub fn rustc_attr() {
}

#[cold]
#[rustc_force_inline]
//~^ ERROR `cold` is incompatible with `#[rustc_force_inline]`
pub fn cold() {
}

#[rustc_force_inline]
//~^ ERROR `variadic` is incompatible with `#[rustc_force_inline]`
pub unsafe extern "C" fn variadic(args: ...) {
}
