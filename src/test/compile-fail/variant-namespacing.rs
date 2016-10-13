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

extern crate variant_namespacing;
pub use variant_namespacing::XE::{XStruct, XTuple, XUnit};
//~^ ERROR `XStruct` has already been defined
//~| ERROR `XTuple` has already been defined
//~| ERROR `XUnit` has already been defined
pub use E::{Struct, Tuple, Unit};
//~^ ERROR `Struct` has already been defined
//~| ERROR `Tuple` has already been defined
//~| ERROR `Unit` has already been defined

fn main() {}
