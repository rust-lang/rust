#![no_std]

//@ eq .index[] | select(.name == "MyEnum").attrs | ., ["#[non_exhaustive]"]
#[non_exhaustive]
pub enum MyEnum {
    First,
}

pub enum NonExhaustiveVariant {
    //@ eq .index[] | select(.name == "Variant").attrs | ., ["#[non_exhaustive]"]
    #[non_exhaustive]
    Variant(i64),
}

//@ eq .index[] | select(.name == "MyStruct").attrs | ., ["#[non_exhaustive]"]
#[non_exhaustive]
pub struct MyStruct {
    pub x: i64,
}
