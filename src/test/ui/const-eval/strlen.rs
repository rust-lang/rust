// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

#![feature(const_str_len, const_str_as_bytes)]

#![crate_type = "lib"]

const S: &str = "foo";
pub const B: &[u8] = S.as_bytes();

pub fn foo() -> [u8; S.len()] {
    let mut buf = [0; S.len()];
    for (i, &c) in S.as_bytes().iter().enumerate() {
        buf[i] = c;
    }
    buf
}
