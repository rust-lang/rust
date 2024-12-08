//@ edition: 2024
//@ run-pass
#![feature(gen_blocks)]
#![feature(optimize_attribute)]
#![feature(stmt_expr_attributes)]
#![feature(async_iterator)]
#![allow(dead_code)]

// make sure that other attributes e.g. `optimize` can be applied to gen blocks and functions

fn main() { }

fn optimize_gen_block() -> impl Iterator<Item = ()> {
    #[optimize(speed)]
    gen { yield (); }
}

#[optimize(speed)]
gen fn optimize_gen_fn() -> i32 {
    yield 1;
    yield 2;
    yield 3;
}

#[optimize(speed)]
async gen fn optimize_async_gen_fn() -> i32 {
    yield 1;
    yield 2;
    yield 3;
}

use std::async_iter::AsyncIterator;

pub fn deduce() -> impl AsyncIterator<Item = ()> {
    #[optimize(size)]
    async gen {
        yield ();
    }
}
