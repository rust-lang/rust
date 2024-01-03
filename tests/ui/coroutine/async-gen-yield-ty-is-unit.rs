// compile-flags: --edition 2024 -Zunstable-options
// check-pass

#![feature(async_stream, gen_blocks, noop_waker)]

use std::{stream::Stream, pin::pin, task::{Context, Waker}};

async gen fn gen_fn() -> &'static str {
    yield "hello"
}

pub fn main() {
    let stream = pin!(gen_fn());
    let waker = Waker::noop();
    let ctx = &mut Context::from_waker(&waker);
    stream.poll_next(ctx);
}
