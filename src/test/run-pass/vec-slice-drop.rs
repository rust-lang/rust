// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(managed_boxes, unsafe_destructor)]

use std::cell::Cell;
use std::gc::{Gc, GC};

// Make sure that destructors get run on slice literals
struct foo {
    x: Gc<Cell<int>>,
}

#[unsafe_destructor]
impl Drop for foo {
    fn drop(&mut self) {
        self.x.set(self.x.get() + 1);
    }
}

fn foo(x: Gc<Cell<int>>) -> foo {
    foo {
        x: x
    }
}

pub fn main() {
    let x = box(GC) Cell::new(0);
    {
        let l = &[foo(x)];
        assert_eq!(l[0].x.get(), 0);
    }
    assert_eq!(x.get(), 1);
}
