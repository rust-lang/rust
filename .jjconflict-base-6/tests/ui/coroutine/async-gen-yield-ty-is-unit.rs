//@ edition: 2024
//@ check-pass

#![feature(async_iterator, gen_blocks)]

use std::{async_iter::AsyncIterator, pin::pin, task::{Context, Waker}};

async gen fn gen_fn() -> &'static str {
    yield "hello"
}

pub fn main() {
    let async_iterator = pin!(gen_fn());
    let ctx = &mut Context::from_waker(Waker::noop());
    async_iterator.poll_next(ctx);
}
