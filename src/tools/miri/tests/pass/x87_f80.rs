//@only-target: x86_64 i686
#![feature(complex_numbers, x87_f80)]

#[cfg(target_arch = "x86")]
use std::arch::x86::x87_f80;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::x87_f80;
use std::num::Complex;

static STATIC: x87_f80 = x87_f80::from_ne_bytes([0xAA; 10]);
static STATIC_AGGREGATE: (u8, x87_f80, i32) = (3, x87_f80::from_ne_bytes([0xBB; 10]), 4);
static STATIC_ARRAY: [x87_f80; 3] = [
    x87_f80::from_ne_bytes([0x11; 10]),
    x87_f80::from_ne_bytes([0x22; 10]),
    x87_f80::from_ne_bytes([0x33; 10]),
];

fn identity_rust(x: x87_f80) -> x87_f80 {
    x
}

extern "C" fn identity_c(x: x87_f80) -> x87_f80 {
    x
}

extern "C" fn complex_identity_c(x: Complex<x87_f80>) -> Complex<x87_f80> {
    x
}

fn main() {
    assert_eq!(STATIC.to_ne_bytes(), [0xAA; 10]);
    assert_eq!(STATIC_AGGREGATE.1.to_ne_bytes(), [0xBB; 10]);
    assert_eq!(STATIC_ARRAY[1].to_ne_bytes(), [0x22; 10]);

    let mut m = std::mem::MaybeUninit::<x87_f80>::uninit();
    unsafe { m.as_mut_ptr().cast::<[u8; 10]>().write([0xCC; 10]) };
    let i = unsafe { m.assume_init() };
    assert_eq!(i.to_ne_bytes(), [0xCC; 10]);

    assert_eq!(identity_rust(i).to_ne_bytes(), [0xCC; 10]);
    assert_eq!(identity_c(i).to_ne_bytes(), [0xCC; 10]);

    let Complex { re, im } = complex_identity_c(Complex::new(
        x87_f80::from_ne_bytes([0xDD; 10]),
        x87_f80::from_ne_bytes([0xEE; 10]),
    ));
    assert_eq!(re.to_ne_bytes(), [0xDD; 10]);
    assert_eq!(im.to_ne_bytes(), [0xEE; 10]);
}
