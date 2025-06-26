#![no_std]

//@ jq .index[] | select(.name == "MyEnum").attrs == ["#[non_exhaustive]"]
#[non_exhaustive]
pub enum MyEnum {
    First,
}

pub enum NonExhaustiveVariant {
    //@ jq .index[] | select(.name == "Variant").attrs == ["#[non_exhaustive]"]
    #[non_exhaustive]
    Variant(i64),
}

//@ jq .index[] | select(.name == "MyStruct").attrs == ["#[non_exhaustive]"]
#[non_exhaustive]
pub struct MyStruct {
    pub x: i64,
}
