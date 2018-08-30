// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// check that casts are not being treated as lexprs.

fn main() {
    let mut a = 0i32;
    let b = &(a as i32);
    a = 1;
    assert_ne!(&a as *const i32, b as *const i32);
    assert_eq!(*b, 0);

    assert_eq!(issue_36936(), 1);
}


struct A(u32);

impl Drop for A {
    fn drop(&mut self) {
        self.0 = 0;
    }
}

fn issue_36936() -> u32 {
    let a = &(A(1) as A);
    a.0
}
