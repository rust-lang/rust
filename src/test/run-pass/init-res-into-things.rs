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

use std::cell::Cell;
use std::gc::{Gc, GC};

// Resources can't be copied, but storing into data structures counts
// as a move unless the stored thing is used afterwards.

struct r {
    i: Gc<Cell<int>>,
}

struct Box { x: r }

#[unsafe_destructor]
impl Drop for r {
    fn drop(&mut self) {
        self.i.set(self.i.get() + 1)
    }
}

fn r(i: Gc<Cell<int>>) -> r {
    r {
        i: i
    }
}

fn test_box() {
    let i = box(GC) Cell::new(0);
    {
        let _a = box(GC) r(i);
    }
    assert_eq!(i.get(), 1);
}

fn test_rec() {
    let i = box(GC) Cell::new(0);
    {
        let _a = Box {x: r(i)};
    }
    assert_eq!(i.get(), 1);
}

fn test_tag() {
    enum t {
        t0(r),
    }

    let i = box(GC) Cell::new(0);
    {
        let _a = t0(r(i));
    }
    assert_eq!(i.get(), 1);
}

fn test_tup() {
    let i = box(GC) Cell::new(0);
    {
        let _a = (r(i), 0);
    }
    assert_eq!(i.get(), 1);
}

fn test_unique() {
    let i = box(GC) Cell::new(0);
    {
        let _a = box r(i);
    }
    assert_eq!(i.get(), 1);
}

fn test_box_rec() {
    let i = box(GC) Cell::new(0);
    {
        let _a = box(GC) Box {
            x: r(i)
        };
    }
    assert_eq!(i.get(), 1);
}

pub fn main() {
    test_box();
    test_rec();
    test_tag();
    test_tup();
    test_unique();
    test_box_rec();
}
