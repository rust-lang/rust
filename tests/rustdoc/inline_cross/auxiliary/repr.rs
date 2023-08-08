#![feature(repr_simd)]

#[repr(C, align(8))]
pub struct ReprC {
    field: u8,
}
#[repr(simd, packed(2))]
pub struct ReprSimd {
    field: u8,
}
#[repr(transparent)]
pub struct ReprTransparent {
    field: u8,
}
#[repr(isize)]
pub enum ReprIsize {
    Bla,
}
#[repr(u8)]
pub enum ReprU8 {
    Bla,
}
