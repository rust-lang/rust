// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test struct inheritance on structs from another crate.
#![feature(struct_inherit)]

// aux-build:inherit_struct_lib.rs
extern crate inherit_struct_lib;

struct S3 : inherit_struct_lib::S1; //~ ERROR super-struct is defined in a different crate

pub fn main() {
}
