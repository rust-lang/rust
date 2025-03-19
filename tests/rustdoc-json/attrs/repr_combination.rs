#![no_std]

// Combinations of `#[repr(..)]` attributes.

//@ is "$.index[*][?(@.name=='ReprCI8')].attrs" '["#[attr = Repr([ReprC, ReprInt(SignedInt(I8))])]\n"]'
#[repr(C, i8)]
pub enum ReprCI8 {
    First,
}

//@ is "$.index[*][?(@.name=='SeparateReprCI16')].attrs" '["#[attr = Repr([ReprC, ReprInt(SignedInt(I16))])]\n"]'
#[repr(C)]
#[repr(i16)]
pub enum SeparateReprCI16 {
    First,
}

//@ is "$.index[*][?(@.name=='ReversedReprCUsize')].attrs" '["#[attr = Repr([ReprInt(UnsignedInt(Usize)), ReprC])]\n"]'
#[repr(usize, C)]
pub enum ReversedReprCUsize {
    First,
}

//@ is "$.index[*][?(@.name=='ReprCPacked')].attrs" '["#[attr = Repr([ReprC, ReprPacked(Align(1 bytes))])]\n"]'
#[repr(C, packed)]
pub struct ReprCPacked {
    a: i8,
    b: i64,
}

//@ is "$.index[*][?(@.name=='SeparateReprCPacked')].attrs" '["#[attr = Repr([ReprC, ReprPacked(Align(2 bytes))])]\n"]'
#[repr(C)]
#[repr(packed(2))]
pub struct SeparateReprCPacked {
    a: i8,
    b: i64,
}

//@ is "$.index[*][?(@.name=='ReversedReprCPacked')].attrs" '["#[attr = Repr([ReprPacked(Align(2 bytes)), ReprC])]\n"]'
#[repr(packed(2), C)]
pub struct ReversedReprCPacked {
    a: i8,
    b: i64,
}

//@ is "$.index[*][?(@.name=='ReprCAlign')].attrs" '["#[attr = Repr([ReprC, ReprAlign(Align(16 bytes))])]\n"]'
#[repr(C, align(16))]
pub struct ReprCAlign {
    a: i8,
    b: i64,
}

//@ is "$.index[*][?(@.name=='SeparateReprCAlign')].attrs" '["#[attr = Repr([ReprC, ReprAlign(Align(2 bytes))])]\n"]'
#[repr(C)]
#[repr(align(2))]
pub struct SeparateReprCAlign {
    a: i8,
    b: i64,
}

//@ is "$.index[*][?(@.name=='ReversedReprCAlign')].attrs" '["#[attr = Repr([ReprAlign(Align(2 bytes)), ReprC])]\n"]'
#[repr(align(2), C)]
pub struct ReversedReprCAlign {
    a: i8,
    b: i64,
}

//@ is "$.index[*][?(@.name=='AlignedExplicitRepr')].attrs" '["#[attr = Repr([ReprC, ReprAlign(Align(16 bytes)), ReprInt(SignedInt(Isize))])]\n"]'
#[repr(C, align(16), isize)]
pub enum AlignedExplicitRepr {
    First,
}

//@ is "$.index[*][?(@.name=='ReorderedAlignedExplicitRepr')].attrs" '["#[attr = Repr([ReprInt(SignedInt(Isize)), ReprC, ReprAlign(Align(16 bytes))])]\n"]'
#[repr(isize, C, align(16))]
pub enum ReorderedAlignedExplicitRepr {
    First,
}
