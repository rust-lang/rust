#![no_std]
#![deny(warnings)]

#[inline(never)]
#[no_mangle]
pub fn wrapping_external_fn(a: u32) -> u32 {
    a.wrapping_mul(a)
}

#[inline(never)]
#[no_mangle]
pub fn panicking_external_fn(a: u32) -> u32 {
    a * a
}
