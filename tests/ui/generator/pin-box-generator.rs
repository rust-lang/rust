// run-pass

#![feature(coroutines, coroutine_trait)]

use std::ops::Coroutine;

fn assert_coroutine<G: Coroutine>(_: G) {
}

fn main() {
    assert_coroutine(static || yield);
    assert_coroutine(Box::pin(static || yield));
}
