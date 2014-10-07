// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:xcrate_unit_struct.rs
extern crate xcrate_unit_struct;

const s1: xcrate_unit_struct::Struct = xcrate_unit_struct::Struct;
static s2: xcrate_unit_struct::Unit = xcrate_unit_struct::UnitVariant;
static s3: xcrate_unit_struct::Unit =
                xcrate_unit_struct::Argument(xcrate_unit_struct::Struct);
static s4: xcrate_unit_struct::Unit = xcrate_unit_struct::Argument(s1);
static s5: xcrate_unit_struct::TupleStruct = xcrate_unit_struct::TupleStruct(20, "foo");

fn f1(_: xcrate_unit_struct::Struct) {}
fn f2(_: xcrate_unit_struct::Unit) {}
fn f3(_: xcrate_unit_struct::TupleStruct) {}

pub fn main() {
    f1(xcrate_unit_struct::Struct);
    f2(xcrate_unit_struct::UnitVariant);
    f2(xcrate_unit_struct::Argument(xcrate_unit_struct::Struct));
    f3(xcrate_unit_struct::TupleStruct(10, "bar"));

    f1(s1);
    f2(s2);
    f2(s3);
    f2(s4);
    f3(s5);
}
