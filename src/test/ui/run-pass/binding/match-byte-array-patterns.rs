// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(slice_patterns)]

fn main() {
    let buf = &[0u8; 4];
    match buf {
        &[0, 1, 0, 0] => unimplemented!(),
        b"true" => unimplemented!(),
        _ => {}
    }

    match buf {
        b"true" => unimplemented!(),
        &[0, 1, 0, 0] => unimplemented!(),
        _ => {}
    }

    match buf {
        b"true" => unimplemented!(),
        &[0, x, 0, 0] => assert_eq!(x, 0),
        _ => unimplemented!(),
    }

    let buf: &[u8] = buf;

    match buf {
        &[0, 1, 0, 0] => unimplemented!(),
        &[_] => unimplemented!(),
        &[_, _, _, _, _, ..] => unimplemented!(),
        b"true" => unimplemented!(),
        _ => {}
    }

    match buf {
        b"true" => unimplemented!(),
        &[0, 1, 0, 0] => unimplemented!(),
        _ => {}
    }

    match buf {
        b"true" => unimplemented!(),
        &[0, x, 0, 0] => assert_eq!(x, 0),
        _ => unimplemented!(),
    }
}
