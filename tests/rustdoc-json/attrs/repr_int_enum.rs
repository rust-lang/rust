#![no_std]

//@ is "$.index[?(@.name=='I8')].attrs" '[{"content": "#[repr(i8)]", "is_inner": false}]'
#[repr(i8)]
pub enum I8 {
    First,
}

//@ is "$.index[?(@.name=='I32')].attrs" '[{"content": "#[repr(i32)]", "is_inner": false}]'
#[repr(i32)]
pub enum I32 {
    First,
}

//@ is "$.index[?(@.name=='Usize')].attrs" '[{"content": "#[repr(usize)]", "is_inner": false}]'
#[repr(usize)]
pub enum Usize {
    First,
}
