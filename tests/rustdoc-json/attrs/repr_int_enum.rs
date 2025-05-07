#![no_std]

//@ is "$.index[?(@.name=='I8')].attrs" '["#[repr(i8)]"]'
#[repr(i8)]
pub enum I8 {
    First,
}

//@ is "$.index[?(@.name=='I32')].attrs" '["#[repr(i32)]"]'
#[repr(i32)]
pub enum I32 {
    First,
}

//@ is "$.index[?(@.name=='Usize')].attrs" '["#[repr(usize)]"]'
#[repr(usize)]
pub enum Usize {
    First,
}
