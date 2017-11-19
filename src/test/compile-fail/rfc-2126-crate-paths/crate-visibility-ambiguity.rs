// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(crate_in_paths)]
#![feature(crate_visibility_modifier)]

mod m {
    pub struct Z;
    pub struct S1(crate (::m::Z)); // OK
    pub struct S2(::crate ::m::Z); // OK
    pub struct S3(crate ::m::Z); //~ ERROR `crate` can only be used in absolute paths
}

fn main() {
    crate struct S; // OK (item in statement position)
}
