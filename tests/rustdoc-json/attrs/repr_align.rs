#![no_std]

//@ is "$.index[?(@.name=='Aligned')].attrs" '[{"content": "#[repr(align(4))]", "is_inner": false}]'
#[repr(align(4))]
pub struct Aligned {
    a: i8,
    b: i64,
}
