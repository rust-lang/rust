// This demonstrates a proposed alternate or additional option of having yield in postfix position.
//@ edition: 2024
//@ pp-exact

#![feature(gen_blocks, coroutines, coroutine_trait, yield_expr,
stmt_expr_attributes)]

use std::ops::{Coroutine, CoroutineState};
use std::pin::pin;

fn main() {
    let mut gn = gen { yield 1; 2.yield; (1 + 2).yield; };

    let mut coro =
        pin!(#[coroutine] |_: i32| { let x = 1.yield; (x + 2).yield; });
}
