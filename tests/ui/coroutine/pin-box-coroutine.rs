//@ run-pass

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::Coroutine;

fn assert_coroutine<G: Coroutine>(_: G) {
}

fn main() {
    assert_coroutine(#[coroutine] static || yield);
    assert_coroutine(Box::pin(#[coroutine] static || yield));
}
