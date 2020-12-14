#![no_std]
#![feature(repr_simd, platform_intrinsics, link_llvm_intrinsics, simd_ffi, min_const_generics)]
#![warn(missing_docs)]
//! Portable SIMD module.

#[macro_use]
mod macros;

mod fmt;
mod intrinsics;
mod ops;
mod round;

mod masks;
pub use masks::*;

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

//mod vectors_mask8;
//pub use vectors_mask8::*;
//mod vectors_mask16;
//pub use vectors_mask16::*;
//mod vectors_mask32;
//pub use vectors_mask32::*;
//mod vectors_mask64;
//pub use vectors_mask64::*;
//mod vectors_mask128;
//pub use vectors_mask128::*;
//mod vectors_masksize;
//pub use vectors_masksize::*;
