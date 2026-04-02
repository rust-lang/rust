//@ run-pass
//! Test using `#[splat]` on tuple arguments of simple functions.

#![allow(incomplete_features)]
#![feature(splat)]

fn tuple_args(#[splat] (_a, _b): (u32, i8)) {}

fn main() {
    tuple_args(1, 2);
    // FIXME(splat): should splatted functions be callable with tupled and un-tupled arguments?
    // Add a tupled test for each call if they are.
    //tuple_args((1, 2));
    tuple_args(1u32, 2i8);
}
