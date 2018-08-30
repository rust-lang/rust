// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unboxed_closures)]
#![feature(fn_traits)]

struct Test;

impl FnOnce<(u32, u32)> for Test {
    type Output = u32;

    extern "rust-call" fn call_once(self, (a, b): (u32, u32)) -> u32 {
        a + b
    }
}

fn main() {
    assert_eq!(Test(1u32, 2u32), 3u32);
}
