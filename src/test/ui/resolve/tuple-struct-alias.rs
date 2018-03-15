// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct S(u8, u16);
type A = S;

impl S {
    fn f() {
        let s = Self(0, 1);
        match s {
            Self(..) => {}
        }
    }
}

fn main() {
    let s = A(0, 1);
    match s {
        A(..) => {}
    }
}
