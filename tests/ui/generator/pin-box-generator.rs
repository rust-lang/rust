// run-pass

#![feature(generators, generator_trait)]

use std::ops::Coroutine;

fn assert_generator<G: Coroutine>(_: G) {
}

fn main() {
    assert_generator(static || yield);
    assert_generator(Box::pin(static || yield));
}
