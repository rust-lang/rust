// compile-flags: --edition 2024 -Zunstable-options
// check-pass

#![feature(async_stream, gen_blocks)]

use std::stream::Stream;

fn deduce() -> impl Stream<Item = ()> {
    async gen {
        yield Default::default();
    }
}

fn main() {}
