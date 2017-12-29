// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(advanced_slice_patterns, slice_patterns)]
#![deny(unreachable_patterns)]

fn main() {
    let buf = &[0, 1, 2, 3];

    match buf {
        b"AAAA" => {},
        &[0x41, 0x41, 0x41, 0x41] => {} //~ ERROR unreachable pattern
        _ => {}
    }

    match buf {
        &[0x41, 0x41, 0x41, 0x41] => {}
        b"AAAA" => {}, //~ ERROR unreachable pattern
        _ => {}
    }

    match buf {
        &[_, 0x41, 0x41, 0x41] => {},
        b"AAAA" => {}, //~ ERROR unreachable pattern
        _ => {}
    }

    match buf {
        &[0x41, .., 0x41] => {}
        b"AAAA" => {}, //~ ERROR unreachable pattern
        _ => {}
    }

    let buf: &[u8] = buf;

    match buf {
        b"AAAA" => {},
        &[0x41, 0x41, 0x41, 0x41] => {} //~ ERROR unreachable pattern
        _ => {}
    }

    match buf {
        &[0x41, 0x41, 0x41, 0x41] => {}
        b"AAAA" => {}, //~ ERROR unreachable pattern
        _ => {}
    }

    match buf {
        &[_, 0x41, 0x41, 0x41] => {},
        b"AAAA" => {}, //~ ERROR unreachable pattern
        _ => {}
    }

    match buf {
        &[0x41, .., 0x41] => {}
        b"AAAA" => {}, //~ ERROR unreachable pattern
        _ => {}
    }
}
