//@ run-pass
//! Test using `#[splat]` on tuple trait arguments of generic functions.

#![allow(incomplete_features)]
#![feature(splat)]
#![feature(tuple_trait)]

fn splat_generic_tuple<T: std::marker::Tuple>(#[splat] _t: T) {}

fn main() {
    // Calling with un-splatted arguments might look like it works, but the actual generic type is
    // a tuple inside another tuple. Aren't generics great?
    splat_generic_tuple((1, 2));
    splat_generic_tuple((1u32, 2i8));

    // Generic tuple trait implementers are resolved during caller typeck.
    splat_generic_tuple::<(u32, i8)>(1u32, 2i8);
    splat_generic_tuple(1u32, 2i8);
    splat_generic_tuple(1, 2);

    splat_generic_tuple::<()>();
    splat_generic_tuple();
}
