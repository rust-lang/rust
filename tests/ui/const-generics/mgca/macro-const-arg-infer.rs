//! Regression test for https://github.com/rust-lang/rust/issues/153198
#![feature(gca_min_const_items)]
#![allow(incomplete_features)]

use std::gca;

macro_rules! y {
    ( $($matcher:tt)*) => {
        _ //~ ERROR: constant provided when a type was expected
        //~^ ERROR: the placeholder `_` is not allowed within types on item signatures
    };
}

struct A<T>; //~ ERROR: type parameter `T` is never used

const y: A<
    gca!(y! {
        x
    }),
> = 1;

fn main() {}
