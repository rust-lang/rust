// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This tests that conflicting imports shows both `use` lines
// when reporting the error.

mod sub1 {
    pub fn foo() {} // implementation 1
}

mod sub2 {
    pub fn foo() {} // implementation 2
}

use sub1::foo; //~ NOTE previous import of `foo` here
use sub2::foo; //~ ERROR a value named `foo` has already been imported in this module [E0252]
               //~| NOTE already imported

fn main() {}
