// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:xcrate_unit_struct.rs

// Make sure that when we have cross-crate unit structs we don't accidentally
// make values out of cross-crate structs that aren't unit.

extern mod xcrate_unit_struct;

fn main() {
    let _ = xcrate_unit_struct::StructWithFields; //~ ERROR: unresolved name
    let _ = xcrate_unit_struct::Struct;
}
