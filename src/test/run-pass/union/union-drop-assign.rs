// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Drop works for union itself.

#![feature(untagged_unions)]

struct S;

union U {
    a: S
}

impl Drop for S {
    fn drop(&mut self) {
        unsafe { CHECK += 10; }
    }
}

impl Drop for U {
    fn drop(&mut self) {
        unsafe { CHECK += 1; }
    }
}

static mut CHECK: u8 = 0;

fn main() {
    unsafe {
        let mut u = U { a: S };
        assert_eq!(CHECK, 0);
        u = U { a: S };
        assert_eq!(CHECK, 1); // union itself is assigned, union is dropped, field is not dropped
        u.a = S;
        assert_eq!(CHECK, 11); // union field is assigned, field is dropped
    }
}
