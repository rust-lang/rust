#![crate_type = "lib"]
#![no_std]

#[inline(never)]
pub fn callee(x: u64) -> u64 {
    x.wrapping_mul(3).wrapping_add(7)
}

pub fn caller(a: u64, b: u64) -> u64 {
    callee(a) + callee(b)
}
