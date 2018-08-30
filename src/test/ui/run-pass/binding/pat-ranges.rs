// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Parsing of range patterns

const NUM1: i32 = 10;

mod m {
    pub const NUM2: i32 = 16;
}

fn main() {
    if let NUM1 ... m::NUM2 = 10 {} else { panic!() }
    if let ::NUM1 ... ::m::NUM2 = 11 {} else { panic!() }
    if let -13 ... -10 = 12 { panic!() } else {}
}
