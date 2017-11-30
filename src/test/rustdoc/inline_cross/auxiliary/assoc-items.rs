// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
