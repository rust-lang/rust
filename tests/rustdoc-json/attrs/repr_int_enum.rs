#![no_std]

//@ is "$.index[*][?(@.name=='I8')].attrs" '["#[attr = Repr([ReprInt(SignedInt(I8))])]\n"]'
#[repr(i8)]
pub enum I8 {
    First,
}

//@ is "$.index[*][?(@.name=='I32')].attrs" '["#[attr = Repr([ReprInt(SignedInt(I32))])]\n"]'
#[repr(i32)]
pub enum I32 {
    First,
}

//@ is "$.index[*][?(@.name=='Usize')].attrs" '["#[attr = Repr([ReprInt(UnsignedInt(Usize))])]\n"]'
#[repr(usize)]
pub enum Usize {
    First,
}
