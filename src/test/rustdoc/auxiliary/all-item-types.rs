// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(extern_types)]

pub mod foo_mod {}
extern "C" {
    pub fn foo_ffn();
    pub static FOO_FSTATIC: FooStruct;
    pub type FooFType;
}
pub fn foo_fn() {}
pub trait FooTrait {}
pub struct FooStruct;
pub enum FooEnum {}
pub union FooUnion {
    x: (),
}
pub type FooType = FooStruct;
pub static FOO_STATIC: FooStruct = FooStruct;
pub const FOO_CONSTANT: FooStruct = FooStruct;
#[macro_export]
macro_rules! foo_macro {
    () => ();
}
