#![no_std]

// Note the normalization:
// `#[repr(packed)]` in source becomes `#[repr(packed(1))]` in rustdoc JSON.
//
//@ jq .index[] | select(.name == "Packed").attrs == ["#[repr(packed(1))]"]
#[repr(packed)]
pub struct Packed {
    a: i8,
    b: i64,
}

//@ jq .index[] | select(.name == "PackedAligned").attrs == ["#[repr(packed(4))]"]
#[repr(packed(4))]
pub struct PackedAligned {
    a: i8,
    b: i64,
}
