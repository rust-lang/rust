//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ [next] compile-flags: -Znext-solver
//@ check-pass

// A regression test for https://github.com/rust-lang/rust/issues/151300

#![feature(transmutability)]
use std::mem::TransmuteFrom;

pub fn is_maybe_transmutable<Src, Dst>()
where
    Dst: TransmuteFrom<Src>,
{
}

fn function_with_generic<T>() {
    is_maybe_transmutable::<(), ()>();
}

fn main() {}
