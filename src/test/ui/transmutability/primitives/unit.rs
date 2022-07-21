//! The unit type, `()`, should be one byte.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code)]

mod assert {
    use std::mem::BikeshedIntrinsicFrom;

    pub fn is_transmutable<Src, Dst, Context>()
    where
        Dst: BikeshedIntrinsicFrom<Src, Context, true, true, true, true>
    {}
}

#[repr(C)]
struct Zst;

fn should_have_correct_size() {
    struct Context;
    assert::is_transmutable::<(), Zst, Context>();
    assert::is_transmutable::<Zst, (), Context>();
    assert::is_transmutable::<(), u8, Context>(); //~ ERROR cannot be safely transmuted
}
