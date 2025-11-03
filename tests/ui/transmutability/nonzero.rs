#![feature(transmutability)]
#![feature(rustc_attrs)]
#![allow(dead_code)]

mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn is_transmutable_assume_nothing<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, { Assume::NOTHING }>,
    {
    }

    pub fn is_transmutable_assume_safety<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, { Assume::SAFETY }>,
    {
    }
}

fn main() {
    // FIXME: Replace this with `core::num::NonZeroU8` once we support it.
    #[rustc_layout_scalar_valid_range_start(1)]
    #[repr(transparent)]
    struct NonZeroU8(u8);

    // FIXME: Replace this with `core::num::NonZeroU16` once we support it.
    #[rustc_layout_scalar_valid_range_start(1)]
    #[repr(transparent)]
    struct NonZeroU16(u16);

    assert::is_transmutable_assume_safety::<u8, NonZeroU8>(); //~ ERROR: cannot be safely transmuted
    assert::is_transmutable_assume_safety::<NonZeroU8, u8>();

    assert::is_transmutable_assume_nothing::<NonZeroU8, NonZeroU8>();  //~ ERROR: cannot be safely transmuted
    assert::is_transmutable_assume_safety::<NonZeroU8, NonZeroU8>();

    assert::is_transmutable_assume_safety::<u16, NonZeroU16>(); //~ ERROR: cannot be safely transmuted
    assert::is_transmutable_assume_safety::<NonZeroU16, u16>();

    assert::is_transmutable_assume_nothing::<NonZeroU16, NonZeroU16>();  //~ ERROR: cannot be safely transmuted
    assert::is_transmutable_assume_safety::<NonZeroU16, NonZeroU16>();
}
