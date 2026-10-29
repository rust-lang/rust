//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)

#![feature(coroutines, coroutine_trait)]

use std::ops::Coroutine;

pub fn example() -> impl Coroutine {
    #[coroutine]
    || yield &1
}

fn main() {}
