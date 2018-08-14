// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deny(unused_unsafe)]
fn main() {
    let mut v = Vec::<i32>::with_capacity(24);

    unsafe {
        let f = |v: &mut Vec<_>| {
            unsafe { //~ ERROR unnecessary `unsafe`
                v.set_len(24);
                |w: &mut Vec<u32>| { unsafe { //~ ERROR unnecessary `unsafe`
                    w.set_len(32);
                } };
            }
            |x: &mut Vec<u32>| { unsafe { //~ ERROR unnecessary `unsafe`
                x.set_len(40);
            } };
        };

        v.set_len(0);
        f(&mut v);
    }

    |y: &mut Vec<u32>| { unsafe {
        y.set_len(48);
    } };
}
