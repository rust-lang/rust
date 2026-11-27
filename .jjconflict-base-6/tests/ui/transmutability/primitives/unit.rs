//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

//! The unit type, `()`, should be one byte.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code)]

mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, {
            Assume::ALIGNMENT
                .and(Assume::LIFETIMES)
                .and(Assume::SAFETY)
                .and(Assume::VALIDITY)
        }>
    {}
}

#[repr(C)]
struct Zst;

fn should_have_correct_size() {
    assert::is_transmutable::<(), Zst>();
    assert::is_transmutable::<Zst, ()>();
    assert::is_transmutable::<(), u8>(); //~ ERROR cannot be safely transmuted
}
