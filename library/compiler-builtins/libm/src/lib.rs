#![deny(warnings)]
#![no_std]

mod fabsf;
mod fmodf;
mod powf;
mod scalbnf;
mod sqrtf;

pub use fabsf::fabsf;
pub use fmodf::fmodf;
pub use powf::powf;
pub use scalbnf::scalbnf;
pub use sqrtf::sqrtf;

/// Approximate equality with 1 ULP of tolerance
#[doc(hidden)]
pub fn _eqf(a: u32, b: u32) -> bool {
    (a as i32).wrapping_sub(b as i32).abs() <= 1
}

fn isnanf(x: f32) -> bool {
    x.to_bits() & 0x7fffffff > 0x7f800000
}
