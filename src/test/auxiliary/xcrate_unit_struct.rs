// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "lib"]

// used by the rpass test

pub struct Struct;

impl Copy for Struct {}

pub enum Unit {
    UnitVariant,
    Argument(Struct)
}

impl Copy for Unit {}

pub struct TupleStruct(pub uint, pub &'static str);

impl Copy for TupleStruct {}

// used by the cfail test

pub struct StructWithFields {
    foo: int,
}

impl Copy for StructWithFields {}

pub enum EnumWithVariants {
    EnumVariant,
    EnumVariantArg(int)
}

impl Copy for EnumWithVariants {}

