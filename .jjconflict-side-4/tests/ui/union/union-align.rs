//@ run-pass

#![allow(dead_code)]

use std::mem::{size_of, size_of_val, align_of, align_of_val};

#[repr(align(16))]
pub union U16 {
    a: u8,
    b: u32
}

fn main() {
    assert_eq!(align_of::<U16>(), 16);
    assert_eq!(size_of::<U16>(), 16);
    let u = U16 { a: 10 };
    unsafe {
        assert_eq!(align_of_val(&u.a), 1);
        assert_eq!(size_of_val(&u.a), 1);
        assert_eq!(u.a, 10);
    }

    let u = U16 { b: 11 };
    unsafe {
        assert_eq!(align_of_val(&u.b), 4);
        assert_eq!(size_of_val(&u.b), 4);
        assert_eq!(u.b, 11);
    }

    hybrid::check_hybrid();
}

mod hybrid {
    use std::mem::{size_of, align_of};

    #[repr(align(16))]
    #[derive(Copy, Clone)]
    struct S1 {
        a: u16,
        b: u8,
    }

    #[repr(align(32))]
    union U {
        s: S1,
        c: u16,
    }

    #[repr(align(64))]
    struct S2 {
        d: u8,
        u: U,
    }

    pub fn check_hybrid() {
        assert_eq!(align_of::<S1>(), 16);
        assert_eq!(size_of::<S1>(), 16);
        assert_eq!(align_of::<U>(), 32);
        assert_eq!(size_of::<U>(), 32);
        assert_eq!(align_of::<S2>(), 64);
        assert_eq!(size_of::<S2>(), 64);
    }
}
