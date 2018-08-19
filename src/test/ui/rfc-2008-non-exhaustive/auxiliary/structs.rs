// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(non_exhaustive)]

#[non_exhaustive]
pub struct NormalStruct {
    pub first_field: u16,
    pub second_field: u16,
}

#[non_exhaustive]
pub struct UnitStruct;

#[non_exhaustive]
pub struct TupleStruct(pub u16, pub u16);

#[derive(Debug)]
#[non_exhaustive]
pub struct FunctionalRecord {
    pub first_field: u16,
    pub second_field: u16,
    pub third_field: bool
}

impl Default for FunctionalRecord {
    fn default() -> FunctionalRecord {
        FunctionalRecord { first_field: 640, second_field: 480, third_field: false }
    }
}
