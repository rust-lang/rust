#![no_std]

// Combinations of `#[repr(..)]` attributes.
// Rustdoc JSON emits normalized output, regardless of the original source.

//@ eq .index[] | select(.name == "ReprCI8").attrs | ., ["#[repr(C, i8)]"]
#[repr(C, i8)]
pub enum ReprCI8 {
    First,
}

//@ eq .index[] | select(.name == "SeparateReprCI16").attrs | ., ["#[repr(C, i16)]"]
#[repr(C)]
#[repr(i16)]
pub enum SeparateReprCI16 {
    First,
}

//@ eq .index[] | select(.name == "ReversedReprCUsize").attrs | ., ["#[repr(C, usize)]"]
#[repr(usize, C)]
pub enum ReversedReprCUsize {
    First,
}

//@ eq .index[] | select(.name == "ReprCPacked").attrs | ., ["#[repr(C, packed(1))]"]
#[repr(C, packed)]
pub struct ReprCPacked {
    a: i8,
    b: i64,
}

//@ eq .index[] | select(.name == "SeparateReprCPacked").attrs | ., ["#[repr(C, packed(2))]"]
#[repr(C)]
#[repr(packed(2))]
pub struct SeparateReprCPacked {
    a: i8,
    b: i64,
}

//@ eq .index[] | select(.name == "ReversedReprCPacked").attrs | ., ["#[repr(C, packed(2))]"]
#[repr(packed(2), C)]
pub struct ReversedReprCPacked {
    a: i8,
    b: i64,
}

//@ eq .index[] | select(.name == "ReprCAlign").attrs | ., ["#[repr(C, align(16))]"]
#[repr(C, align(16))]
pub struct ReprCAlign {
    a: i8,
    b: i64,
}

//@ eq .index[] | select(.name == "SeparateReprCAlign").attrs | ., ["#[repr(C, align(2))]"]
#[repr(C)]
#[repr(align(2))]
pub struct SeparateReprCAlign {
    a: i8,
    b: i64,
}

//@ eq .index[] | select(.name == "ReversedReprCAlign").attrs | ., ["#[repr(C, align(2))]"]
#[repr(align(2), C)]
pub struct ReversedReprCAlign {
    a: i8,
    b: i64,
}

//@ eq .index[] | select(.name == "AlignedExplicitRepr").attrs | ., ["#[repr(C, align(16), isize)]"]
#[repr(C, align(16), isize)]
pub enum AlignedExplicitRepr {
    First,
}

//@ eq .index[] | select(.name == "ReorderedAlignedExplicitRepr").attrs | ., ["#[repr(C, align(16), isize)]"]
#[repr(isize, C, align(16))]
pub enum ReorderedAlignedExplicitRepr {
    First,
}

//@ eq .index[] | select(.name == "Transparent").attrs | ., ["#[repr(transparent)]"]
#[repr(transparent)]
pub struct Transparent(i64);
