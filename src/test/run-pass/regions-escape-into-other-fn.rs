// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(managed_boxes)]

use std::gc::GC;

fn foo<'r>(x: &'r uint) -> &'r uint { x }
fn bar(x: &uint) -> uint { *x }

pub fn main() {
    let p = box(GC) 3u;
    assert_eq!(bar(foo(p)), 3);
}
