//@ run-pass
//! Test using `#[splat]` on tuple trait arguments of generic functions.

#![allow(incomplete_features)]
#![feature(splat)]
#![feature(tuple_trait)]

fn splat_generic_tuple<T: std::marker::Tuple>(#[splat] _t: T) {}

fn main() {
    // FIXME(splat): should splatted functions be callable with tupled and un-tupled arguments?
    // Add a tupled test for each call if they are.
    //splat_generic_tuple((1, 2));

    // Generic tuple trait implementers are resolved during caller typeck.
    splat_generic_tuple::<(u32, i8)>(1u32, 2i8);
    splat_generic_tuple(1u32, 2i8);
    splat_generic_tuple(1, 2);

    splat_generic_tuple::<()>();
    splat_generic_tuple();
}
