//@ revisions: current next
//@[next] compile-flags: -Znext-solver

//! The unit type, `()`, should be one byte.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code)]

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<Src, {
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
