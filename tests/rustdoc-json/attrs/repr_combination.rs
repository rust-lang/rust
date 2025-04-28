#![no_std]

// Combinations of `#[repr(..)]` attributes.
// Rustdoc JSON emits normalized output, regardless of the original source.

//@ is "$.index[?(@.name=='ReprCI8')].attrs" '[{"content": "#[repr(C, i8)]", "is_inner": false}]'
#[repr(C, i8)]
pub enum ReprCI8 {
    First,
}

//@ is "$.index[?(@.name=='SeparateReprCI16')].attrs" '[{"content": "#[repr(C, i16)]", "is_inner": false}]'
#[repr(C)]
#[repr(i16)]
pub enum SeparateReprCI16 {
    First,
}

//@ is "$.index[?(@.name=='ReversedReprCUsize')].attrs" '[{"content": "#[repr(C, usize)]", "is_inner": false}]'
#[repr(usize, C)]
pub enum ReversedReprCUsize {
    First,
}

//@ is "$.index[?(@.name=='ReprCPacked')].attrs" '[{"content": "#[repr(C, packed(1))]", "is_inner": false}]'
#[repr(C, packed)]
pub struct ReprCPacked {
    a: i8,
    b: i64,
}

//@ is "$.index[?(@.name=='SeparateReprCPacked')].attrs" '[{"content": "#[repr(C, packed(2))]", "is_inner": false}]'
#[repr(C)]
#[repr(packed(2))]
pub struct SeparateReprCPacked {
    a: i8,
    b: i64,
}

//@ is "$.index[?(@.name=='ReversedReprCPacked')].attrs" '[{"content": "#[repr(C, packed(2))]", "is_inner": false}]'
#[repr(packed(2), C)]
pub struct ReversedReprCPacked {
    a: i8,
    b: i64,
}

//@ is "$.index[?(@.name=='ReprCAlign')].attrs" '[{"content": "#[repr(C, align(16))]", "is_inner": false}]'
#[repr(C, align(16))]
pub struct ReprCAlign {
    a: i8,
    b: i64,
}

//@ is "$.index[?(@.name=='SeparateReprCAlign')].attrs" '[{"content": "#[repr(C, align(2))]", "is_inner": false}]'
#[repr(C)]
#[repr(align(2))]
pub struct SeparateReprCAlign {
    a: i8,
    b: i64,
}

//@ is "$.index[?(@.name=='ReversedReprCAlign')].attrs" '[{"content": "#[repr(C, align(2))]", "is_inner": false}]'
#[repr(align(2), C)]
pub struct ReversedReprCAlign {
    a: i8,
    b: i64,
}

//@ is "$.index[?(@.name=='AlignedExplicitRepr')].attrs" '[{"content": "#[repr(C, align(16), isize)]", "is_inner": false}]'
#[repr(C, align(16), isize)]
pub enum AlignedExplicitRepr {
    First,
}

//@ is "$.index[?(@.name=='ReorderedAlignedExplicitRepr')].attrs" '[{"content": "#[repr(C, align(16), isize)]", "is_inner": false}]'
#[repr(isize, C, align(16))]
pub enum ReorderedAlignedExplicitRepr {
    First,
}
