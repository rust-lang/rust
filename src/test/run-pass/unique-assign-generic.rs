// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::gc::GC;

fn f<T>(t: T) -> T {
    let t1 = t;
    t1
}

pub fn main() {
    let t = f(box 100i);
    assert_eq!(t, box 100i);
    let t = f(box box(GC) vec!(100i));
    assert_eq!(t, box box(GC) vec!(100i));
}
