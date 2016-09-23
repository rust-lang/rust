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

use std::mem::{size_of, size_of_val, align_of, align_of_val};

struct S {
    a: u16,
    b: [u8; 3],
}

#[repr(packed)]
struct Sp {
    a: u16,
    b: [u8; 3],
}

union U {
    a: u16,
    b: [u8; 3],
}

#[repr(packed)]
union Up {
    a: u16,
    b: [u8; 3],
}

const CS: S = S { a: 0, b: [0, 0, 0] };
const CSP: Sp = Sp { a: 0, b: [0, 0, 0] };
const CU: U = U { b: [0, 0, 0] };
const CUP: Up = Up { b: [0, 0, 0] };

fn main() {
    let s = S { a: 0, b: [0, 0, 0] };
    assert_eq!(size_of::<S>(), 6);
    assert_eq!(size_of_val(&s), 6);
    assert_eq!(size_of_val(&CS), 6);
    assert_eq!(align_of::<S>(), 2);
    assert_eq!(align_of_val(&s), 2);
    assert_eq!(align_of_val(&CS), 2);

    let sp = Sp { a: 0, b: [0, 0, 0] };
    assert_eq!(size_of::<Sp>(), 5);
    assert_eq!(size_of_val(&sp), 5);
    assert_eq!(size_of_val(&CSP), 5);
    assert_eq!(align_of::<Sp>(), 1);
    assert_eq!(align_of_val(&sp), 1);
    assert_eq!(align_of_val(&CSP), 1);

    let u = U { b: [0, 0, 0] };
    assert_eq!(size_of::<U>(), 4);
    assert_eq!(size_of_val(&u), 4);
    assert_eq!(size_of_val(&CU), 4);
    assert_eq!(align_of::<U>(), 2);
    assert_eq!(align_of_val(&u), 2);
    assert_eq!(align_of_val(&CU), 2);

    let up = Up { b: [0, 0, 0] };
    assert_eq!(size_of::<Up>(), 3);
    assert_eq!(size_of_val(&up), 3);
    assert_eq!(size_of_val(&CUP), 3);
    assert_eq!(align_of::<Up>(), 1);
    assert_eq!(align_of_val(&up), 1);
    assert_eq!(align_of_val(&CUP), 1);

    hybrid::check_hybrid();
}

mod hybrid {
    use std::mem::size_of;

    #[repr(packed)]
    struct S1 {
        a: u16,
        b: u8,
    }

    #[repr(packed)]
    union U {
        s: S1,
        c: u16,
    }

    #[repr(packed)]
    struct S2 {
        d: u8,
        u: U,
    }

    pub fn check_hybrid() {
        assert_eq!(size_of::<S1>(), 3);
        assert_eq!(size_of::<U>(), 3);
        assert_eq!(size_of::<S2>(), 4);
    }
}
