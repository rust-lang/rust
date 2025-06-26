#![no_std]

//@ eq .index[] | select(.name == "ReprCStruct").attrs | ., ["#[repr(C)]"]
#[repr(C)]
pub struct ReprCStruct(pub i64);

//@ eq .index[] | select(.name == "ReprCEnum").attrs | ., ["#[repr(C)]"]
#[repr(C)]
pub enum ReprCEnum {
    First,
}

//@ eq .index[] | select(.name == "ReprCUnion").attrs | ., ["#[repr(C)]"]
#[repr(C)]
pub union ReprCUnion {
    pub left: i64,
    pub right: u64,
}
