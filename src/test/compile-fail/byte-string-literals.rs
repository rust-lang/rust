// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// ignore-tidy-tab

static FOO: &'static [u8] = b"\f";  //~ ERROR unknown byte escape

pub fn main() {
    b"\f";  //~ ERROR unknown byte escape
    b"\x0Z";  //~ ERROR illegal character in numeric character escape: Z
    b"é";  //~ ERROR byte constant must be ASCII
    b"a  //~ ERROR unterminated double quote byte string
}


