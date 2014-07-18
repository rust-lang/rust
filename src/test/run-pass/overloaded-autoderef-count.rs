// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::Cell;
use std::ops::{Deref, DerefMut};

#[deriving(PartialEq)]
struct DerefCounter<T> {
    count_imm: Cell<uint>,
    count_mut: uint,
    value: T
}

impl<T> DerefCounter<T> {
    fn new(value: T) -> DerefCounter<T> {
        DerefCounter {
            count_imm: Cell::new(0),
            count_mut: 0,
            value: value
        }
    }

    fn counts(&self) -> (uint, uint) {
        (self.count_imm.get(), self.count_mut)
    }
}

impl<T> Deref<T> for DerefCounter<T> {
    fn deref(&self) -> &T {
        self.count_imm.set(self.count_imm.get() + 1);
        &self.value
    }
}

impl<T> DerefMut<T> for DerefCounter<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.count_mut += 1;
        &mut self.value
    }
}

#[deriving(PartialEq, Show)]
struct Point {
    x: int,
    y: int
}

impl Point {
    fn get(&self) -> (int, int) {
        (self.x, self.y)
    }
}

pub fn main() {
    let mut p = DerefCounter::new(Point {x: 0, y: 0});

    let _ = p.x;
    assert_eq!(p.counts(), (1, 0));

    let _ = &p.x;
    assert_eq!(p.counts(), (2, 0));

    let _ = &mut p.y;
    assert_eq!(p.counts(), (2, 1));

    p.x += 3;
    assert_eq!(p.counts(), (2, 2));

    p.get();
    assert_eq!(p.counts(), (2, 3));

    // Check the final state.
    assert_eq!(*p, Point {x: 3, y: 0});
}
