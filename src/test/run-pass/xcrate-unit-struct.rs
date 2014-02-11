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
// ignore-fast
extern mod xcrate_unit_struct;

static s1: xcrate_unit_struct::Struct = xcrate_unit_struct::Struct;
static s2: xcrate_unit_struct::Unit = xcrate_unit_struct::Unit;
static s3: xcrate_unit_struct::Unit =
                xcrate_unit_struct::Argument(xcrate_unit_struct::Struct);
static s4: xcrate_unit_struct::Unit = xcrate_unit_struct::Argument(s1);

fn f1(_: xcrate_unit_struct::Struct) {}
fn f2(_: xcrate_unit_struct::Unit) {}

pub fn main() {
    f1(xcrate_unit_struct::Struct);
    f2(xcrate_unit_struct::Unit);
    f2(xcrate_unit_struct::Argument(xcrate_unit_struct::Struct));

    f1(s1);
    f2(s2);
    f2(s3);
    f2(s4);
}
