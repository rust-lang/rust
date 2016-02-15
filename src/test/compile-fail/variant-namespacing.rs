// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:variant-namespacing.rs

extern crate variant_namespacing;
pub use variant_namespacing::XE::{XStruct, XTuple, XUnit};
//~^ ERROR import `XStruct` conflicts with type in this module
//~| ERROR import `XStruct` conflicts with value in this module
//~| ERROR import `XTuple` conflicts with type in this module
//~| ERROR import `XTuple` conflicts with value in this module
//~| ERROR import `XUnit` conflicts with type in this module
//~| ERROR import `XUnit` conflicts with value in this module
pub use E::{Struct, Tuple, Unit};
//~^ ERROR import `Struct` conflicts with type in this module
//~| ERROR import `Struct` conflicts with value in this module
//~| ERROR import `Tuple` conflicts with type in this module
//~| ERROR import `Tuple` conflicts with value in this module
//~| ERROR import `Unit` conflicts with type in this module
//~| ERROR import `Unit` conflicts with value in this module

enum E {
    Struct { a: u8 },
    Tuple(u8),
    Unit,
}

type Struct = u8;
type Tuple = u8;
type Unit = u8;
type XStruct = u8;
type XTuple = u8;
type XUnit = u8;

const Struct: u8 = 0;
const Tuple: u8 = 0;
const Unit: u8 = 0;
const XStruct: u8 = 0;
const XTuple: u8 = 0;
const XUnit: u8 = 0;

fn main() {}
