#![no_std]

//@ count "$.index[?(@.name=='Aligned')].attrs[*]" 1
//@ is "$.index[?(@.name=='Aligned')].attrs[*].repr.align" 4
#[repr(align(4))]
pub struct Aligned {
    a: i8,
    b: i64,
}
