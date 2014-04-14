// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(struct_variant, macro_rules)]

#[deriving(Show)]
struct Unit;

#[deriving(Show)]
struct Tuple(int, uint);

#[deriving(Show)]
struct Struct { x: int, y: uint }

#[deriving(Show)]
enum Enum {
    Nullary,
    Variant(int, uint),
    StructVariant { x: int, y : uint }
}

macro_rules! t {
    ($x:expr, $expected:expr) => {
        assert_eq!(format!("{}", $x), $expected.to_owned())
    }
}

pub fn main() {
    t!(Unit, "Unit");
    t!(Tuple(1, 2), "Tuple(1, 2)");
    t!(Struct { x: 1, y: 2 }, "Struct { x: 1, y: 2 }");
    t!(Nullary, "Nullary");
    t!(Variant(1, 2), "Variant(1, 2)");
    t!(StructVariant { x: 1, y: 2 }, "StructVariant { x: 1, y: 2 }");
}
