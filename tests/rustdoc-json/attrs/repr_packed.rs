#![no_std]

// Note the normalization:
// `#[repr(packed)]` in has the implict "1" in rustdoc JSON.

//@ is "$.index[*][?(@.name=='Packed')].attrs" '["#[attr = Repr([ReprPacked(Align(1 bytes))])]\n"]'
#[repr(packed)]
pub struct Packed {
    a: i8,
    b: i64,
}

//@ is "$.index[*][?(@.name=='PackedAligned')].attrs" '["#[attr = Repr([ReprPacked(Align(4 bytes))])]\n"]'
#[repr(packed(4))]
pub struct PackedAligned {
    a: i8,
    b: i64,
}
