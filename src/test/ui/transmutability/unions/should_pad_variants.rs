//! The variants of a union must be padded with uninit bytes such that they have
//! the same length (in bytes).

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code)]

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};

    pub fn is_transmutable<Src, Dst, Context>()
    where
        Dst: BikeshedIntrinsicFrom<Src, Context, {
            Assume::ALIGNMENT
                .and(Assume::LIFETIMES)
                .and(Assume::SAFETY)
                .and(Assume::VALIDITY)
        }>
    {}
}

#[derive(Clone, Copy)]
#[repr(C)] struct Zst;

#[derive(Clone, Copy)]
#[repr(u8)] enum V0 { V = 0 }

#[derive(Clone, Copy)]
#[repr(u8)] enum V2 { V = 2 }

#[repr(C)]
union Lopsided {
    smol: Zst,
    lorg: V0,
}

#[repr(C)] struct Src(V0, Zst, V2);
#[repr(C)] struct Dst(V0, Lopsided, V2);

fn should_pad_variants() {
    struct Context;
    // If the implementation (incorrectly) fails to pad `Lopsided::smol` with
    // an uninitialized byte, this transmutation might be (wrongly) accepted:
    assert::is_transmutable::<Src, Dst, Context>(); //~ ERROR cannot be safely transmuted
}
