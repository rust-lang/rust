// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use foo::baz;
use bar::baz; //~ ERROR a module named `baz` has already been imported

use foo::Quux;
use bar::Quux; //~ ERROR a trait named `Quux` has already been imported

use foo::blah;
use bar::blah; //~ ERROR a type named `blah` has already been imported

use foo::WOMP;
use bar::WOMP; //~ ERROR a value named `WOMP` has already been imported

fn main() {}

mod foo {
    pub mod baz {}
    pub trait Quux { }
    pub type blah = (f64, u32);
    pub const WOMP: u8 = 5;
}

mod bar {
    pub mod baz {}
    pub type Quux = i32;
    pub struct blah { x: i8 }
    pub const WOMP: i8 = -5;
}
