//! Checks that closures and rust-call functions can't be splatted.
//! This should be rejected until we decide on sensible semantics.

#![feature(splat, unboxed_closures, tuple_trait)]
#![expect(incomplete_features)]

use std::marker::Tuple;

trait Trait: Tuple + Sized {
    extern "rust-call" fn method(#[splat] self: Self);
    //~^ ERROR: `#[splat]` is not allowed in the arguments of functions with the `rust-call` ABI
}

impl Trait for (i32, i64) {
    extern "rust-call" fn method(#[splat] self: Self) {
        //~^ ERROR: `#[splat]` is not allowed in the arguments of functions with the `rust-call` ABI
        println!("{self:?}");
    }
}

fn main() {
    (|#[splat] x: i32| {
        //~^ ERROR `#[splat]` is not allowed on closure arguments
        println!("{x}");
    })(1);

    (1_i32, 2_i64).method();
    Trait::method(3_i32, 4_i64);
    //~^ ERROR functions with the "rust-call" ABI must take a single non-self tuple argument
}
