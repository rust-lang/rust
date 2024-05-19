//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ run-pass

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]
#![allow(dropping_copy_types)]

use std::marker::PhantomPinned;

fn assert_unpin<G: Unpin>(_: G) {
}

fn main() {
    // Even though this coroutine holds a `PhantomPinned` in its environment, it
    // remains `Unpin`.
    assert_unpin(#[coroutine] || {
        let pinned = PhantomPinned;
        yield;
        drop(pinned);
    });
}
