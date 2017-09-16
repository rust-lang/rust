// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Inner {
    type T;
}

impl<'a> Inner for &'a i32 {
    type T = i32;
}

fn f<'a>(x: &'a i32) -> <&'a i32 as Inner>::T {
    *x
}

fn main() {}
