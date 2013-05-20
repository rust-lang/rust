// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// exec-env:RUST_POISON_ON_FREE=1

fn borrow(x: &int, f: &fn(x: &int)) {
    let before = *x;
    f(x);
    let after = *x;
    assert_eq!(before, after);
}

struct F { f: ~int }

pub fn main() {
    let mut x = @F {f: ~3};
    do borrow((*x).f) |b_x| {
        assert_eq!(*b_x, 3);
        assert_eq!(ptr::to_unsafe_ptr(&(*x.f)), ptr::to_unsafe_ptr(&(*b_x)));
        x = @F {f: ~4};

        debug!("ptr::to_unsafe_ptr(*b_x) = %x",
               ptr::to_unsafe_ptr(&(*b_x)) as uint);
        assert_eq!(*b_x, 3);
        assert!(ptr::to_unsafe_ptr(&(*x.f)) != ptr::to_unsafe_ptr(&(*b_x)));
    }
}
