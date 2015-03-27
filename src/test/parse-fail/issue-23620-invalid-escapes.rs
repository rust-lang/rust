// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let _ = b"\u{a66e}";
    //~^ ERROR unicode escape sequences cannot be used as a byte or in a byte string

    let _ = b'\u{a66e}';
    //~^ ERROR unicode escape sequences cannot be used as a byte or in a byte string

    let _ = b'\u';
    //~^ ERROR unknown byte escape: u

    let _ = b'\x5';
    //~^ ERROR numeric character escape is too short

    let _ = b'\xxy';
    //~^ ERROR illegal character in numeric character escape: x
    //~^^ ERROR illegal character in numeric character escape: y

    let _ = '\x5';
    //~^ ERROR numeric character escape is too short

    let _ = '\xxy';
    //~^ ERROR illegal character in numeric character escape: x
    //~^^ ERROR illegal character in numeric character escape: y

    let _ = b"\u{a4a4} \xf \u";
    //~^ ERROR unicode escape sequences cannot be used as a byte or in a byte string
    //~^^ ERROR illegal character in numeric character escape:
    //~^^^ ERROR unknown byte escape: u

    let _ = "\u{ffffff} \xf \u";
    //~^ ERROR illegal unicode character escape
    //~^^ ERROR illegal character in numeric character escape:
    //~^^^ ERROR form of character escape may only be used with characters in the range [\x00-\x7f]
    //~^^^^ ERROR unknown character escape: u
}
