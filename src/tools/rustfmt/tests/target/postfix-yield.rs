// This demonstrates a proposed alternate or additional option of having yield in postfix position.
//@ edition: 2024

#![feature(gen_blocks, coroutines, coroutine_trait, yield_expr)]

use std::ops::{Coroutine, CoroutineState};
use std::pin::pin;

fn main() {
    let mut coro = pin!(
        #[coroutine]
        |_: i32| {
            let x = 1.yield;
            (x + 2).await;
        }
    );
}
