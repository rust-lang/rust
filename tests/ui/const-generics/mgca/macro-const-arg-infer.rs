//! Regression test for https://github.com/rust-lang/rust/issues/153198
#![feature(min_generic_const_args)]
#![allow(incomplete_features)]
macro_rules! y {
    ( $($matcher:tt)*) => {
        _ //~ ERROR: the placeholder `_` is not allowed within types on item signatures
    };
}

struct A<T>; //~ ERROR: type parameter `T` is never used

const y: A<
    {
        y! {
            x
        }
    },
> = 1; //~ ERROR: mismatched types

fn main() {}
