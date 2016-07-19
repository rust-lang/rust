// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(deprecated)]

#[deprecated(since = "1.0.0", note = "text")]
pub fn deprecated() {}
#[deprecated(since = "1.0.0", note = "text")]
pub fn deprecated_text() {}

pub struct MethodTester;

impl MethodTester {
    #[deprecated(since = "1.0.0", note = "text")]
    pub fn method_deprecated(&self) {}
    #[deprecated(since = "1.0.0", note = "text")]
    pub fn method_deprecated_text(&self) {}
}

pub trait Trait {
    #[deprecated(since = "1.0.0", note = "text")]
    fn trait_deprecated(&self) {}
    #[deprecated(since = "1.0.0", note = "text")]
    fn trait_deprecated_text(&self) {}
}

#[deprecated(since = "1.0.0", note = "text")]
pub trait DeprecatedTrait { fn dummy(&self) { } }

impl Trait for MethodTester {}

#[deprecated(since = "1.0.0", note = "text")]
pub struct DeprecatedStruct {
    pub i: isize
}

#[deprecated(since = "1.0.0", note = "text")]
pub struct DeprecatedUnitStruct;

pub enum Enum {
    #[deprecated(since = "1.0.0", note = "text")]
    DeprecatedVariant,
}

#[deprecated(since = "1.0.0", note = "text")]
pub struct DeprecatedTupleStruct(pub isize);

pub struct Stable {
    #[deprecated(since = "1.0.0", note = "text")]
    pub override2: u8,
}

pub struct Stable2(pub u8, pub u8, #[deprecated(since = "1.0.0", note = "text")] pub u8);

#[deprecated(since = "1.0.0", note = "text")]
pub struct Deprecated {
    pub inherit: u8,
}

#[deprecated(since = "1.0.0", note = "text")]
pub struct Deprecated2(pub u8,
                       pub u8,
                       pub u8);

#[deprecated(since = "1.0.0", note = "text")]
pub mod deprecated_mod {
    pub fn deprecated() {}
}

#[macro_export]
macro_rules! macro_test {
    () => (deprecated());
}

#[macro_export]
macro_rules! macro_test_arg {
    ($func:expr) => ($func);
}

#[macro_export]
macro_rules! macro_test_arg_nested {
    ($func:ident) => (macro_test_arg!($func()));
}
