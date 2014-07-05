// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// Test to make sure the destructors run in the right order.
// Each destructor sets it's tag in the corresponding entry
// in ORDER matching up to when it ran.
// Correct order is: matched, inner, outer

static mut ORDER: [uint, ..3] = [0, 0, 0];
static mut INDEX: uint = 0;

struct A;
impl Drop for A {
    fn drop(&mut self) {
        unsafe {
            ORDER[INDEX] = 1;
            INDEX = INDEX + 1;
        }
    }
}

struct B;
impl Drop for B {
    fn drop(&mut self) {
        unsafe {
            ORDER[INDEX] = 2;
            INDEX = INDEX + 1;
        }
    }
}

struct C;
impl Drop for C {
    fn drop(&mut self) {
        unsafe {
            ORDER[INDEX] = 3;
            INDEX = INDEX + 1;
        }
    }
}

fn main() {
    {
        let matched = A;
        let _outer = C;
        {
            match matched {
                _s => {}
            }
            let _inner = B;
        }
    }
    unsafe {
        assert_eq!(&[1, 2, 3], ORDER.as_slice());
    }
}
