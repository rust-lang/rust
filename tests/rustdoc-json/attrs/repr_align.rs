#![no_std]

//@ is "$.index[*][?(@.name=='Aligned')].attrs" '["#[attr = Repr([ReprAlign(Align(4 bytes))])]\n"]'
#[repr(align(4))]
pub struct Aligned {
    a: i8,
    b: i64,
}
