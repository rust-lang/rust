// gate-test-required_inlining
//@ compile-flags: --crate-type=lib
#![allow(internal_features)]
#![feature(rustc_attrs)]

#[rustc_no_mir_inline]
#[inline(required)]
//~^ ERROR invalid argument
//~^^ HELP valid inline arguments are `always` and `never`
pub fn bare_required() {
}

#[rustc_no_mir_inline]
#[inline(required("justification"))]
//~^ ERROR invalid argument
//~^^ HELP valid inline arguments are `always` and `never`
pub fn justified_required() {
}

#[rustc_no_mir_inline]
#[inline(must)]
//~^ ERROR invalid argument
//~^^ HELP valid inline arguments are `always` and `never`
pub fn bare_must() {
}

#[rustc_no_mir_inline]
#[inline(must("justification"))]
//~^ ERROR invalid argument
//~^^ HELP valid inline arguments are `always` and `never`
pub fn justified_must() {
}
