//@ run-pass

#![allow(unused_imports)]

//@ aux-build:union.rs

extern crate union;
use std::mem::{size_of, align_of, zeroed};

union U {
    a: u8,
    b: u16
}

fn local() {
    assert_eq!(size_of::<U>(), 2);
    assert_eq!(align_of::<U>(), 2);

    let u = U { a: 10 };
    unsafe {
        assert_eq!(u.a, 10);
        let U { a } = u;
        assert_eq!(a, 10);
    }

    let mut w = U { b: 0 };
    unsafe {
        assert_eq!(w.a, 0);
        assert_eq!(w.b, 0);
        w.a = 1;
        assert_eq!(w.a, 1);
        assert_eq!(w.b.to_le(), 1);
    }
}

fn xcrate() {
    assert_eq!(size_of::<union::U>(), 2);
    assert_eq!(align_of::<union::U>(), 2);

    let u = union::U { a: 10 };
    unsafe {
        assert_eq!(u.a, 10);
        let union::U { a } = u;
        assert_eq!(a, 10);
    }

    let mut w = union::U { b: 0 };
    unsafe {
        assert_eq!(w.a, 0);
        assert_eq!(w.b, 0);
        w.a = 1;
        assert_eq!(w.a, 1);
        assert_eq!(w.b.to_le(), 1);
    }
}

fn main() {
    local();
    xcrate();
}
