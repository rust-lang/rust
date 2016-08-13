// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(relaxed_adts)]

struct S(u8, u16);

fn main() {
    let s = S{1: 10, 0: 11};
    match s {
        S{0: a, 1: b, ..} => {
            assert_eq!(a, 11);
            assert_eq!(b, 10);
        }
    }
}
