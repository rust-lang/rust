//! Borrowing from packed arrays with generic length:
//! - allowed if ABI align([T]) == align(T) <= packed alignment
//! - hard error (E0793) otherwise
#![allow(dead_code, unused_variables, unused_mut)]
use std::mem::MaybeUninit;
use std::num::{NonZeroU8, NonZeroU16};

//
// -------- PASS CASES --------
//

mod pass_u8 {
    #[repr(C, packed)]
    pub struct PascalString<const N: usize> {
        len: u8,
        buf: [u8; N],
    }

    pub fn bar<const N: usize>(s: &PascalString<N>) -> &str {
        // should NOT trigger E0793
        std::str::from_utf8(&s.buf[0..s.len as usize]).unwrap()
    }

    pub fn run() {
        let p = PascalString::<10> { len: 3, buf: *b"abc\0\0\0\0\0\0\0" };
        let s = bar(&p);
        assert_eq!(s, "abc");
    }
}

mod pass_i8 {
    #[repr(C, packed)]
    pub struct S<const N: usize> {
        buf: [i8; N],
    }
    pub fn run() {
        let s = S::<4> { buf: [1, 2, 3, 4] };
        let _ok = &s.buf[..]; // no E0793
    }
}

mod pass_nonzero_u8 {
    use super::*;
    #[repr(C, packed)]
    pub struct S<const N: usize> {
        buf: [NonZeroU8; N],
    }
    pub fn run() {
        let s = S::<3> {
            buf: [
                NonZeroU8::new(1).unwrap(),
                NonZeroU8::new(2).unwrap(),
                NonZeroU8::new(3).unwrap(),
            ],
        };
        let _ok = &s.buf[..]; // no E0793
    }
}

mod pass_maybeuninit_u8 {
    use super::*;
    #[repr(C, packed)]
    pub struct S<const N: usize> {
        buf: [MaybeUninit<u8>; N],
    }
    pub fn run() {
        let s = S::<2> { buf: [MaybeUninit::new(1), MaybeUninit::new(2)] };
        let _ok = &s.buf[..]; // no E0793
    }
}

mod pass_transparent_u8 {
    #[repr(transparent)]
    pub struct WrapU8(u8);

    #[repr(C, packed)]
    pub struct S<const N: usize> {
        buf: [WrapU8; N],
    }
    pub fn run() {
        let s = S::<2> { buf: [WrapU8(1), WrapU8(2)] };
        let _ok = &s.buf[..]; // no E0793
    }
}

//
// -------- FAIL CASES (expect E0793) --------
//

mod fail_u16 {
    #[repr(C, packed)]
    pub struct S<const N: usize> {
        buf: [u16; N],
    }
    pub fn run() {
        let s = S::<2> { buf: [1, 2] };
        let _err = &s.buf[..];
        //~^ ERROR: reference to packed field is unaligned
    }
}

mod fail_nonzero_u16 {
    use super::*;
    #[repr(C, packed)]
    pub struct S<const N: usize> {
        buf: [NonZeroU16; N],
    }
    pub fn run() {
        let s = S::<1> { buf: [NonZeroU16::new(1).unwrap()] };
        let _err = &s.buf[..];
        //~^ ERROR: reference to packed field is unaligned
    }
}

mod fail_transparent_u16 {
    #[repr(transparent)]
    pub struct WrapU16(u16);

    #[repr(C, packed)]
    pub struct S<const N: usize> {
        buf: [WrapU16; N],
    }
    pub fn run() {
        let s = S::<1> { buf: [WrapU16(42)] };
        let _err = &s.buf[..];
        //~^ ERROR: reference to packed field is unaligned
    }
}

fn main() {
    // Run pass cases (fail cases only check diagnostics)
    pass_u8::run();
    pass_i8::run();
    pass_nonzero_u8::run();
    pass_maybeuninit_u8::run();
    pass_transparent_u8::run();
}
