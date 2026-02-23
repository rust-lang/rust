#![feature(associated_type_defaults)]

pub struct MyStruct;

impl MyStruct {
    /// docs for PrivateConst
    const PrivateConst: i8 = -123;
    /// docs for PublicConst
    pub const PublicConst: u8 = 123;
    /// docs for private_method
    fn private_method() {}
    /// docs for public_method
    pub fn public_method() {}
}

pub trait MyTrait {
    /// docs for ConstNoDefault
    const ConstNoDefault: i16;
    /// docs for ConstWithDefault
    const ConstWithDefault: u16 = 12345;
    /// docs for TypeNoDefault
    type TypeNoDefault;
    /// docs for TypeWithDefault
    type TypeWithDefault = u32;
    /// docs for method_no_default
    fn method_no_default();
    /// docs for method_with_default
    fn method_with_default() {}
}

impl MyTrait for MyStruct {
    /// dox for ConstNoDefault
    const ConstNoDefault: i16 = -12345;
    /// dox for TypeNoDefault
    type TypeNoDefault = i32;
    /// dox for method_no_default
    fn method_no_default() {}
}
