#![deny(warnings)]
#![no_std]

mod math;

pub use math::*;

/// Approximate equality with 1 ULP of tolerance
#[doc(hidden)]
pub fn _eqf(a: u32, b: u32) -> bool {
    (a as i32).wrapping_sub(b as i32).abs() <= 1
}

#[doc(hidden)]
pub fn _eq(a: u64, b: u64) -> bool {
    (a as i64).wrapping_sub(b as i64).abs() <= 1
}
