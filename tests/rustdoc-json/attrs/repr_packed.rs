#![no_std]

// Note the normalization:
// `#[repr(packed)]` in source becomes `{"repr": {"packed": 1, ...}}` in rustdoc JSON.
//
//@ is "$.index[?(@.name=='Packed')].attrs[*].repr.packed" 1
//@ is "$.index[?(@.name=='Packed')].attrs[*].repr.kind" '"rust"'
#[repr(packed)]
pub struct Packed {
    a: i8,
    b: i64,
}

//@ is "$.index[?(@.name=='PackedAligned')].attrs[*].repr.packed" 4
//@ is "$.index[?(@.name=='PackedAligned')].attrs[*].repr.kind" '"rust"'
#[repr(packed(4))]
pub struct PackedAligned {
    a: i8,
    b: i64,
}
