#![feature(repr_simd)]

#[repr(Rust)]
pub struct ReprRust;

#[repr(C, align(8))] // public
pub struct ReprC {
    pub field: u8,
}

#[repr(C)] // private
pub struct ReprCPrivField {
    private: u8,
    pub public: i8,
}

#[repr(align(4))] // private
pub struct ReprAlignHiddenField {
    #[doc(hidden)]
    pub hidden: i16,
}

#[repr(simd, packed(2))] // public
pub struct ReprSimd {
    pub field: u8,
}

#[repr(transparent)] // public
pub struct ReprTransparent {
    pub field: u8, // non-1-ZST field
}

#[repr(isize)] // public
pub enum ReprIsize {
    Bla,
}

#[repr(u8)] // public
pub enum ReprU8 {
    Bla,
}

#[repr(u32)] // public
pub enum ReprU32 {
    #[doc(hidden)]
    Hidden,
    Public,
}

#[repr(u64)] // private
pub enum ReprU64HiddenVariants {
    #[doc(hidden)]
    A,
    #[doc(hidden)]
    B,
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
