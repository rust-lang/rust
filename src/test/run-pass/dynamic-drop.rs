// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::RefCell;

struct Allocator {
    data: RefCell<Vec<bool>>,
}

impl Drop for Allocator {
    fn drop(&mut self) {
        let data = self.data.borrow();
        if data.iter().any(|d| *d) {
            panic!("missing free: {:?}", data);
        }
    }
}

impl Allocator {
    fn new() -> Self { Allocator { data: RefCell::new(vec![]) } }
    fn alloc(&self) -> Ptr {
        let mut data = self.data.borrow_mut();
        let addr = data.len();
        data.push(true);
        Ptr(addr, self)
    }
}

struct Ptr<'a>(usize, &'a Allocator);
impl<'a> Drop for Ptr<'a> {
    fn drop(&mut self) {
        match self.1.data.borrow_mut()[self.0] {
            false => {
                panic!("double free at index {:?}", self.0)
            }
            ref mut d => *d = false
        }
    }
}

fn dynamic_init(a: &Allocator, c: bool) {
    let _x;
    if c {
        _x = Some(a.alloc());
    }
}

fn dynamic_drop(a: &Allocator, c: bool) -> Option<Ptr> {
    let x = a.alloc();
    if c {
        Some(x)
    } else {
        None
    }
}

fn assignment2(a: &Allocator, c0: bool, c1: bool) {
    let mut _v = a.alloc();
    let mut _w = a.alloc();
    if c0 {
        drop(_v);
    }
    _v = _w;
    if c1 {
        _w = a.alloc();
    }
}

fn assignment1(a: &Allocator, c0: bool) {
    let mut _v = a.alloc();
    let mut _w = a.alloc();
    if c0 {
        drop(_v);
    }
    _v = _w;
}


fn main() {
    let a = Allocator::new();
    dynamic_init(&a, false);
    dynamic_init(&a, true);
    dynamic_drop(&a, false);
    dynamic_drop(&a, true);

    assignment2(&a, false, false);
    assignment2(&a, false, true);
    assignment2(&a, true, false);
    assignment2(&a, true, true);

    assignment1(&a, false);
    assignment1(&a, true);
}
