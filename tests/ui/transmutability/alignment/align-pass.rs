//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] known-bug: trait-system-refactor-initiative#152
//@ check-pass
#![feature(transmutability)]

mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn is_maybe_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, {
            Assume {
                alignment: false,
                lifetimes: false,
                safety: true,
                validity: false,
            }
        }>
    {}
}

fn main() {
    assert::is_maybe_transmutable::<&'static [u16; 0], &'static [u8; 0]>();
}
