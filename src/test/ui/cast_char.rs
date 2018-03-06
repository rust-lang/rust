// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(overflowing_literals)]

fn main() {
    const XYZ: char = 0x1F888 as char;
    //~^ ERROR only u8 can be cast into char
    const XY: char = 129160 as char;
    //~^ ERROR only u8 can be cast into char
    const ZYX: char = '\u{01F888}';
    println!("{}", XYZ);
}
