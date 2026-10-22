#![crate_type = "lib"]
#![no_std]
#![feature(optimize_attribute)]

#[inline(never)]
pub fn callee(x: u64) -> u64 {
    x.wrapping_mul(3).wrapping_add(7)
}

#[optimize(size)]
pub fn caller_size(a: u64, b: u64) -> u64 {
    callee(a) + callee(b)
}

#[optimize(speed)]
pub fn caller_speed(a: u64, b: u64) -> u64 {
    callee(a) + callee(b)
}
