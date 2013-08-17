// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct c1<T> {
    x: T,
}

impl<T> c1<T> {
    pub fn f1(&self, _x: int) {
    }
}

fn c1<T>(x: T) -> c1<T> {
    c1 {
        x: x
    }
}

impl<T> c1<T> {
    pub fn f2(&self, _x: int) {
    }
}


pub fn main() {
    c1::<int>(3).f1(4);
    c1::<int>(3).f2(4);
}
