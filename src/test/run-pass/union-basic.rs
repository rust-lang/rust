// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(untagged_unions)]

use std::mem::{size_of, align_of, zeroed};

union U {
    a: u8,
}

union U64 {
    a: u64,
}

union W {
    a: u8,
    b: u64,
}

#[repr(C)]
union Y {
    f1: u16,
    f2: [u8; 4],
}

fn main() {
    assert_eq!(size_of::<U>(), 1);
    assert_eq!(size_of::<U64>(), 8);
    assert_eq!(size_of::<W>(), 8);
    assert_eq!(align_of::<U>(), 1);
    assert_eq!(align_of::<U64>(), align_of::<u64>());
    assert_eq!(align_of::<W>(), align_of::<u64>());
    assert_eq!(size_of::<Y>(), 4);
    assert_eq!(align_of::<Y>(), 2);

    let u = U { a: 10 };
    unsafe {
        assert_eq!(u.a, 10);
        let U { a } = u;
        assert_eq!(a, 10);
    }

    let mut w = W { b: 0 };
    unsafe {
        assert_eq!(w.a, 0);
        assert_eq!(w.b, 0);
        w.a = 1;
        assert_eq!(w.a, 1);
        assert_eq!(w.b, 1);
    }
}
