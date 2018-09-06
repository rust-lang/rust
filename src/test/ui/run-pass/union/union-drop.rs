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
    a: u8
}

union W {
    a: S,
}

union Y {
    a: S,
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

impl Drop for W {
    fn drop(&mut self) {
        unsafe { CHECK += 1; }
    }
}

static mut CHECK: u8 = 0;

fn main() {
    unsafe {
        assert_eq!(CHECK, 0);
        {
            let u = U { a: 1 };
        }
        assert_eq!(CHECK, 1); // 1, dtor of U is called
        {
            let w = W { a: S };
        }
        assert_eq!(CHECK, 2); // 2, not 11, dtor of S is not called
        {
            let y = Y { a: S };
        }
        assert_eq!(CHECK, 2); // 2, not 12, dtor of S is not called
    }
}
