// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#![feature(lang_items, unboxed_closures, fn_traits)]

struct S3 {
    x: i32,
    y: i32,
}

impl FnOnce<(i32,i32)> for S3 {
    type Output = i32;
    extern "rust-call" fn call_once(self, (z,zz): (i32,i32)) -> i32 {
        self.x * self.y * z * zz
    }
}

fn main() {
    let s = S3 {
        x: 3,
        y: 3,
    };
    let ans = s(3, 1);
    assert_eq!(ans, 27);
}
