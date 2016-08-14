// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensures that destructors are run for expressions of the form "e;" where
// `e` is a type which requires a destructor.


#![allow(path_statements)]

struct A { n: isize }
struct B;

static mut NUM_DROPS: usize = 0;

impl Drop for A {
    fn drop(&mut self) {
        unsafe { NUM_DROPS += 1; }
    }
}

impl Drop for B {
    fn drop(&mut self) {
        unsafe { NUM_DROPS += 1; }
    }
}

fn main() {
    assert_eq!(unsafe { NUM_DROPS }, 0);
    { let _a = A { n: 1 }; }
    assert_eq!(unsafe { NUM_DROPS }, 1);
    { A { n: 3 }; }
    assert_eq!(unsafe { NUM_DROPS }, 2);

    { let _b = B; }
    assert_eq!(unsafe { NUM_DROPS }, 3);
    { B; }
    assert_eq!(unsafe { NUM_DROPS }, 4);
}
