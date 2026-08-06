//! Diagnostics for expressions that contain things that must be a mGCA direct expression, *and*
//! things that must be an anon const, are currently less than ideal. This test merely asserts the
//! current (bad) state of diagnostics, so we can track improvements over time.

#![feature(min_generic_const_args, min_adt_const_params)]
#![allow(incomplete_features)]

fn f<const N: (u32, u32)>() {}

fn g<const N: u32>() {
    f::<{ (N, 1 + 1) }>();
    //~^ ERROR: generic parameters may not be used in const operations
    f::<{ core::direct_const_arg!((N, 1 + 1)) }>();
    //~^ ERROR: complex const arguments must be placed inside of a `const` block
}

fn main() {}
