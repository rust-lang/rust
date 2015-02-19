// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct A { pub i: isize }
pub enum C { pub Variant }      //~ ERROR: unnecessary `pub`

pub trait E {
    pub fn foo(&self) {}         //~ ERROR: unnecessary visibility
}
trait F {
    pub fn foo(&self) {}     //~ ERROR: unnecessary visibility
}

impl E for A {
    pub fn foo(&self) {}             //~ ERROR: unnecessary visibility
}

fn main() {}
