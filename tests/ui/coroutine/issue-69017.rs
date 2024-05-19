// This issue reproduces an ICE on compile
// Fails on 2020-02-08 nightly
// regressed commit: https://github.com/rust-lang/rust/commit/f8fd4624474a68bd26694eff3536b9f3a127b2d3
//
//@ check-pass

#![feature(coroutine_trait)]
#![feature(coroutines)]

use std::ops::Coroutine;

fn gen() -> impl Coroutine<usize> {
    #[coroutine]
    |_: usize| {
        println!("-> {}", yield);
    }
}

fn main() {}
