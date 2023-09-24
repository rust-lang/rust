#![feature(try_trait_v2)]

use std::ops::Try;

fn main() {
    // error for a `Try` type on a non-`Try` fn
    std::fs::File::open("foo")?; //~ ERROR the `?` operator can only

    // a non-`Try` type on a non-`Try` fn
    0i32?; //~ ERROR the `?` operator can only be applied to

    // an unrelated use of `Try`
    try_trait_generic::<i32>(); //~ ERROR the trait bound
}

fn try_trait_generic<T: Try>() -> T {
    // and a non-`Try` object on a `Try` fn.
    0i32?; //~ ERROR the `?` operator can only be applied to values that implement `Try`

    loop {}
}
