// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// More thorough regression test for Issues #30018 and #30822. This
// attempts to explore different ways that array element construction
// (for both scratch arrays and non-scratch ones) interacts with
// breaks in the control-flow, in terms of the order of evaluation of
// the destructors (which may change; see RFC Issue 744) and the
// number of times that the destructor evaluates for each value (which
// should never exceed 1; this latter case is what #30822 is about).

use std::cell::RefCell;

struct D<'a>(&'a RefCell<Vec<i32>>, i32);

impl<'a> Drop for D<'a> {
    fn drop(&mut self) {
        println!("Dropping D({})", self.1);
        (self.0).borrow_mut().push(self.1);
    }
}

fn main() {
    println!("Start");
    break_during_elem();
    break_after_whole();
    println!("Finis");
}

fn break_during_elem() {
    let log = &RefCell::new(Vec::new());

    // CASE 1: Fixed-size array itself is stored in _r slot.
    loop {
        let _r = [D(log, 10),
                  D(log, 11),
                  { D(log, 12); break; },
                  D(log, 13)];
    }
    assert_eq!(&log.borrow()[..], &[12, 11, 10]);
    log.borrow_mut().clear();

    // CASE 2: Slice (borrow of array) is stored in _r slot.
    // This is the case that is actually being reported in #30018.
    loop {
        let _r = &[D(log, 20),
                   D(log, 21),
                   { D(log, 22); break; },
                   D(log, 23)];
    }
    assert_eq!(&log.borrow()[..], &[22, 21, 20]);
    log.borrow_mut().clear();

    // CASE 3: (Borrow of) slice-index of array is stored in _r slot.
    loop {
        let _r = &[D(log, 30),
                  D(log, 31),
                  { D(log, 32); break; },
                  D(log, 33)][..];
    }
    assert_eq!(&log.borrow()[..], &[32, 31, 30]);
    log.borrow_mut().clear();
}

// The purpose of these functions is to test what happens when we
// panic after an array has been constructed in its entirety.
//
// It is meant to act as proof that we still need to continue
// scheduling the destruction of an array even after we've scheduling
// drop for its elements during construction; the latter is tested by
// `fn break_during_elem()`.
fn break_after_whole() {
    let log = &RefCell::new(Vec::new());

    // CASE 1: Fixed-size array itself is stored in _r slot.
    loop {
        let _r = [D(log, 10),
                  D(log, 11),
                  D(log, 12)];
        break;
    }
    assert_eq!(&log.borrow()[..], &[10, 11, 12]);
    log.borrow_mut().clear();

    // CASE 2: Slice (borrow of array) is stored in _r slot.
    loop {
        let _r = &[D(log, 20),
                   D(log, 21),
                   D(log, 22)];
        break;
    }
    assert_eq!(&log.borrow()[..], &[20, 21, 22]);
    log.borrow_mut().clear();

    // CASE 3: (Borrow of) slice-index of array is stored in _r slot.
    loop {
        let _r = &[D(log, 30),
                   D(log, 31),
                   D(log, 32)][..];
        break;
    }
    assert_eq!(&log.borrow()[..], &[30, 31, 32]);
    log.borrow_mut().clear();
}
