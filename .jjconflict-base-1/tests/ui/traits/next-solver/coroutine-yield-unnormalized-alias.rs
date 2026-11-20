//@ check-pass
//@ edition: 2024
//@ compile-flags: -Znext-solver=globally
//! Regression test for #160652. Yielding an item from `impl Iterator` without an
//! explicit `Item` bound used to ICE in NLL type relating: both the MIR `yield_ty`
//! and the yielded local `i` were `<impl Iterator as Iterator>::Item`.

#![feature(coroutines, coroutine_trait)]

use std::ops::Coroutine;

fn iter() -> impl Iterator {
    Some(()).into_iter()
}

fn yield_unnormalized_item() -> impl Coroutine {
    #[coroutine]
    move || {
        for i in iter() {
            yield i
        }
    }
}

fn main() {
    let _ = yield_unnormalized_item();
}
