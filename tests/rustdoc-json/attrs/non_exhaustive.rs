#![no_std]

//@ is "$.index[?(@.name=='MyEnum')].attrs" '["non_exhaustive"]'
#[non_exhaustive]
pub enum MyEnum {
    First,
}

pub enum NonExhaustiveVariant {
    //@ is "$.index[?(@.name=='Variant')].attrs" '["non_exhaustive"]'
    #[non_exhaustive]
    Variant(i64),
}

//@ is "$.index[?(@.name=='MyStruct')].attrs" '["non_exhaustive"]'
#[non_exhaustive]
pub struct MyStruct {
    pub x: i64,
}
