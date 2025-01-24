#![feature(repr_simd)]

#[repr(C, align(8))]
pub struct ReprC {
    field: u8,
}
#[repr(simd, packed(2))]
pub struct ReprSimd {
    field: [u8; 1],
}
#[repr(transparent)]
pub struct ReprTransparent {
    pub field: u8,
}
#[repr(isize)]
pub enum ReprIsize {
    Bla,
}
#[repr(u8)]
pub enum ReprU8 {
    Bla,
}

#[repr(transparent)] // private
pub struct ReprTransparentPrivField {
    field: u32, // non-1-ZST field
}

#[repr(transparent)] // public
pub struct ReprTransparentPriv1ZstFields {
    marker0: Marker,
    pub main: u64, // non-1-ZST field
    marker1: Marker,
}

#[repr(transparent)] // private
pub struct ReprTransparentPrivFieldPub1ZstFields {
    main: [u16; 0], // non-1-ZST field
    pub marker: Marker,
}

pub struct Marker; // 1-ZST
