// Regression test for #166129: an elided lifetime in a fn signature is late-bound, so the type of
// the anon const argument contained an escaping bound region and typeck of the anon const ICEd.

#![feature(generic_const_parameter_types, adt_const_params, unsized_const_params)]
#![expect(incomplete_features)]

use std::marker::PhantomData;

struct S<'a, const N: &'a str>(PhantomData<&'a ()>);

fn issue(_: S<{}>) {}
//~^ ERROR: anonymous constants with lifetimes in their type are not yet supported

fn elided(_: S<{ "a" }>) {}
//~^ ERROR: anonymous constants with lifetimes in their type are not yet supported

fn named<'a>(_: S<'a, { "a" }>) {}
//~^ ERROR: anonymous constants with lifetimes in their type are not yet supported

fn fn_ptr(_: for<'a> fn(S<'a, { "a" }>)) {}
//~^ ERROR: anonymous constants with lifetimes in their type are not yet supported

fn dyn_fn(_: &dyn for<'a> Fn(S<'a, { "a" }>)) {}
//~^ ERROR: anonymous constants with lifetimes in their type are not yet supported

fn main() {}
