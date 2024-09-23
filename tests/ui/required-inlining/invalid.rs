//@ compile-flags: --crate-type=lib
#![allow(internal_features)]
#![feature(rustc_attrs)]
#![feature(required_inlining)]

// Test that invalid required/must inlining attributes error as expected.

#[inline(required = "bar")]
//~^ ERROR invalid argument
//~^^ HELP expected one string argument to `#[inline(required)]`
pub fn reqd1() {
}

#[inline(required(bar, baz))]
//~^ ERROR invalid argument
//~^^ HELP expected one string argument to `#[inline(required)]`
pub fn reqd2() {
}

#[inline(required(2))]
//~^ ERROR invalid argument
//~^^ HELP expected one string argument to `#[inline(required)]`
pub fn reqd3() {
}

#[inline(required = 2)]
//~^ ERROR invalid argument
//~^^ HELP expected one string argument to `#[inline(required)]`
pub fn reqd4() {
}

#[inline(must = "bar")]
//~^ ERROR invalid argument
//~^^ HELP expected one string argument to `#[inline(must)]`
pub fn must1() {
}

#[inline(must(bar, baz))]
//~^ ERROR invalid argument
//~^^ HELP expected one string argument to `#[inline(must)]`
pub fn must2() {
}

#[inline(must(2))]
//~^ ERROR invalid argument
//~^^ HELP expected one string argument to `#[inline(must)]`
pub fn must3() {
}

#[inline(must = 2)]
//~^ ERROR invalid argument
//~^^ HELP expected one string argument to `#[inline(must)]`
pub fn must4() {
}

#[inline(banana)]
//~^ ERROR invalid argument
//~^^ HELP valid inline arguments are `required`, `must`, `always` and `never`
pub fn bare() {
}
