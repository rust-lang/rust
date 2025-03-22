#![no_std]

//@ is "$.index[?(@.name=='ReprCStruct')].attrs" '["#[repr(C)]"]'
#[repr(C)]
pub struct ReprCStruct(pub i64);

//@ is "$.index[?(@.name=='ReprCEnum')].attrs" '["#[repr(C)]"]'
#[repr(C)]
pub enum ReprCEnum {
    First,
}

//@ is "$.index[?(@.name=='ReprCUnion')].attrs" '["#[repr(C)]"]'
#[repr(C)]
pub union ReprCUnion {
    pub left: i64,
    pub right: u64,
}
