//! Regression test for https://github.com/rust-lang/rust/issues/100727.
//! A tuple element of type `!` should be coerced using constraints from `collect`.

//@ check-pass
//@ edition: 2021

#![allow(unreachable_code)]

fn main() {
    let _: Vec<(i32,)> = [(todo!(),)].into_iter().collect();
}
