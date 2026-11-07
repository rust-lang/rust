// https://github.com/rust-lang/rust/issues/115718
#![crate_name = "foo"]

pub trait MyTrait1 {
    fn method_trait_1();
}

pub trait MyTrait2 {
    fn method_trait_2();
}

pub struct MyStruct<T>(T);

impl MyStruct<u32> {
    pub fn method_u32() {}
}

impl MyStruct<u16> {
    pub fn method_u16() {}
}

impl MyTrait1 for MyStruct<u32> {
    fn method_trait_1() {}
}

impl MyTrait2 for MyStruct<u16> {
    fn method_trait_2() {}
}

//@ hasraw 'type.impl/foo/struct.MyStruct.js' 'method_u16'
//@ !hasraw 'type.impl/foo/struct.MyStruct.js' 'method_u32'
//@ !hasraw 'type.impl/foo/struct.MyStruct.js' 'method_trait_1'
//@ hasraw 'type.impl/foo/struct.MyStruct.js' 'method_trait_2'
pub type MyType = MyStruct<u16>;
