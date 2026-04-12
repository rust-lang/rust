// test that reasonable usage of Align works as expected
//@ check-pass
#![feature(align_type)]

use std::mem::Align;

struct ContainsJustAlign {
    align: Align<8>
}

struct WithAlignType {
    a: [u8; 4],
    b: u32,
    c: *const (),

    align: Align<16>
}

#[repr(align(16))]
struct WithReprAlign {
    a: [u8; 4],
    b: u32,
    c: *const (),
}

const XKCD_CERTIFIED_RANDOM_NUMBER: usize = 4;
const MAX_SUPPORTED_ALIGN: usize = 1 << 29;

const _: () = {
    // FIXME: should this fail?
    assert!(size_of::<Align<0>>() == 0);
    assert!(align_of::<Align<0>>() == 1);

    assert!(size_of::<Align<1>>() == 0);
    assert!(align_of::<Align<1>>() == 1);

    assert!(size_of::<Align<64>>() == 0);
    assert!(align_of::<Align<64>>() == 64);

    assert!(size_of::<Align<XKCD_CERTIFIED_RANDOM_NUMBER>>() == 0);
    assert!(align_of::<Align<XKCD_CERTIFIED_RANDOM_NUMBER>>() == XKCD_CERTIFIED_RANDOM_NUMBER);

    assert!(size_of::<Align<MAX_SUPPORTED_ALIGN>>() == 0);
    assert!(align_of::<Align<MAX_SUPPORTED_ALIGN>>() == MAX_SUPPORTED_ALIGN);

    assert!(size_of::<ContainsJustAlign>() == 0);
    assert!(align_of::<ContainsJustAlign>() == 8);

    assert!(size_of::<WithAlignType>() == size_of::<WithReprAlign>());
    assert!(align_of::<WithAlignType>() == size_of::<WithReprAlign>());
};

fn main() {}
