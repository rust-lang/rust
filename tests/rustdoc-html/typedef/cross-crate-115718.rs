//@ aux-build: parent-crate-115718.rs

// https://github.com/rust-lang/rust/issues/115718
#![crate_name = "foo"]

extern crate parent_crate_115718;

use parent_crate_115718::MyStruct;

pub trait MyTrait2 {
    fn method_trait_2();
}

impl MyTrait2 for MyStruct<u16> {
    fn method_trait_2() {}
}

pub trait MyTrait3 {
    fn method_trait_3();
}

impl MyTrait3 for MyType {
    fn method_trait_3() {}
}

//@ hasraw 'type.impl/parent_crate_115718/struct.MyStruct.js' 'method_trait_1'
//@ hasraw 'type.impl/parent_crate_115718/struct.MyStruct.js' 'method_trait_2'
// Avoid duplicating these docs.
//@ !hasraw 'foo/type.MyType.html' 'method_trait_1'
//@ !hasraw 'foo/type.MyType.html' 'method_trait_2'
// The one made directly on the type alias should be attached to the HTML instead.
//@ !hasraw 'type.impl/parent_crate_115718/struct.MyStruct.js' 'method_trait_3'
//@ hasraw 'foo/type.MyType.html' 'method_trait_3'
pub type MyType = MyStruct<u16>;
