#![no_std]

//@ is "$.index[?(@.name=='MyEnum')].attrs" '[{"content": "#[non_exhaustive]", "is_inner": false}]'
#[non_exhaustive]
pub enum MyEnum {
    First,
}

pub enum NonExhaustiveVariant {
    //@ is "$.index[?(@.name=='Variant')].attrs" '[{"content": "#[non_exhaustive]", "is_inner": false}]'
    #[non_exhaustive]
    Variant(i64),
}

//@ is "$.index[?(@.name=='MyStruct')].attrs" '[{"content": "#[non_exhaustive]", "is_inner": false}]'
#[non_exhaustive]
pub struct MyStruct {
    pub x: i64,
}
