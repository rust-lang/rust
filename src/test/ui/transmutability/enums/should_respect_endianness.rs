//! The target endianness should be a consideration in computing the layout of
//! an enum with a multi-byte tag.

#![crate_type = "lib"]
#![feature(arbitrary_enum_discriminant)]
#![feature(transmutability)]
#![allow(dead_code)]

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};
    pub struct Context;

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<Src, Context, {
            Assume::ALIGNMENT
                .and(Assume::LIFETIMES)
                .and(Assume::SAFETY)
                .and(Assume::VALIDITY)
        }>
    {}
}

#[repr(u16)] enum Src { V = 0xCAFE }

#[repr(u8)] enum OxCA { V = 0xCA }
#[repr(u8)] enum OxFE { V = 0xFE }

#[cfg(target_endian = "big")] #[repr(C)] struct Expected(OxCA, OxFE);
#[cfg(target_endian = "big")] #[repr(C)] struct Unexpected(OxFE, OxCA);

#[cfg(target_endian = "little")] #[repr(C)] struct Expected(OxFE, OxCA);
#[cfg(target_endian = "little")] #[repr(C)] struct Unexpected(OxCA, OxFE);

fn should_respect_endianness() {
    assert::is_transmutable::<Src, Expected>();
    assert::is_transmutable::<Src, Unexpected>(); //~ ERROR cannot be safely transmuted
}
