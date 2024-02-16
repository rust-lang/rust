//@ compile-flags: --edition 2024 -Zunstable-options
//@ check-pass

#![feature(async_iterator, gen_blocks)]

use std::async_iter::AsyncIterator;

fn deduce() -> impl AsyncIterator<Item = ()> {
    async gen {
        yield Default::default();
    }
}

fn main() {}
