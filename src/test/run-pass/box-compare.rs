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

pub fn main() {
    assert!((box(GC) 1i < box(GC) 3i));
    assert!((box(GC) box(GC) "hello ".to_string() >
             box(GC) box(GC) "hello".to_string()));
    assert!((box(GC) box(GC) box(GC) "hello".to_string() !=
             box(GC) box(GC) box(GC) "there".to_string()));
}
