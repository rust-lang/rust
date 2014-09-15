// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

static FOO: uint = FOO; //~ ERROR recursive constant

fn main() {
    let _x: [u8, ..FOO]; // caused stack overflow prior to fix
    let _y: uint = 1 + {
        static BAR: uint = BAR; //~ ERROR recursive constant
        let _z: [u8, ..BAR]; // caused stack overflow prior to fix
        1
    };
}
