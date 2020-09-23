#![no_std]
#![feature(repr_simd)]
#![warn(missing_docs)]

#[macro_use]
mod macros;

mod vectors_u8;
pub use vectors_u8::*;
mod vectors_u16;
pub use vectors_u16::*;
mod vectors_u32;
pub use vectors_u32::*;
mod vectors_u64;
pub use vectors_u64::*;
mod vectors_u128;
pub use vectors_u128::*;
mod vectors_usize;
pub use vectors_usize::*;
mod vectors_i8;
pub use vectors_i8::*;
mod vectors_i16;
pub use vectors_i16::*;
mod vectors_i32;
pub use vectors_i32::*;
mod vectors_i64;
pub use vectors_i64::*;
mod vectors_i128;
pub use vectors_i128::*;
mod vectors_isize;
pub use vectors_isize::*;
mod vectors_f32;
pub use vectors_f32::*;
mod vectors_f64;
pub use vectors_f64::*;
