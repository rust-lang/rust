// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

const X1: &'static [u8] = &[b'1'];
const X2: &'static [u8] = b"1";
const X3: &'static [u8; 1] = &[b'1'];
const X4: &'static [u8; 1] = b"1";

static Y1: u8 = X1[0];
static Y2: u8 = X2[0];
static Y3: u8 = X3[0];
static Y4: u8 = X4[0];

fn main() {
    assert_eq!(Y1, Y2);
    assert_eq!(Y1, Y3);
    assert_eq!(Y1, Y4);
}
