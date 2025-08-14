//! This module implements the [WebAssembly `SIMD128` ISA].
//!
//! [WebAssembly `SIMD128` ISA]:
//! https://github.com/WebAssembly/simd/blob/master/proposals/simd/SIMD.md

#![allow(non_camel_case_types)]
#![allow(unused_imports)]

use crate::{core_arch::simd, intrinsics::simd::*, marker::Sized, mem, ptr};

#[cfg(test)]
use stdarch_test::assert_instr;

types! {
    #![stable(feature = "wasm_simd", since = "1.54.0")]

    /// WASM-specific 128-bit wide SIMD vector type.
    ///
    /// This type corresponds to the `v128` type in the [WebAssembly SIMD
    /// proposal](https://github.com/webassembly/simd). This type is 128-bits
    /// large and the meaning of all the bits is defined within the context of
    /// how this value is used.
    ///
    /// This same type is used simultaneously for all 128-bit-wide SIMD types,
    /// for example:
    ///
    /// * sixteen 8-bit integers (both `i8` and `u8`)
    /// * eight 16-bit integers (both `i16` and `u16`)
    /// * four 32-bit integers (both `i32` and `u32`)
    /// * two 64-bit integers (both `i64` and `u64`)
    /// * four 32-bit floats (`f32`)
    /// * two 64-bit floats (`f64`)
    ///
    /// The `v128` type in Rust is intended to be quite analogous to the `v128`
    /// type in WebAssembly. Operations on `v128` can only be performed with the
    /// functions in this module.
    // N.B., internals here are arbitrary.
    pub struct v128(4 x i32);
}

macro_rules! conversions {
    ($(($name:ident = $ty:ty))*) => {
        impl v128 {
            $(
                #[inline(always)]
                pub(crate) fn $name(self) -> $ty {
                    unsafe { mem::transmute(self) }
                }
            )*
        }
        $(
            impl $ty {
                #[inline(always)]
                pub(crate) const fn v128(self) -> v128 {
                    unsafe { mem::transmute(self) }
                }
            }
        )*
    }
}

conversions! {
    (as_u8x16 = simd::u8x16)
    (as_u16x8 = simd::u16x8)
    (as_u32x4 = simd::u32x4)
    (as_u64x2 = simd::u64x2)
    (as_i8x16 = simd::i8x16)
    (as_i16x8 = simd::i16x8)
    (as_i32x4 = simd::i32x4)
    (as_i64x2 = simd::i64x2)
    (as_f32x4 = simd::f32x4)
    (as_f64x2 = simd::f64x2)
}

#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.wasm.swizzle"]
    fn llvm_swizzle(a: simd::i8x16, b: simd::i8x16) -> simd::i8x16;

    #[link_name = "llvm.wasm.bitselect.v16i8"]
    fn llvm_bitselect(a: simd::i8x16, b: simd::i8x16, c: simd::i8x16) -> simd::i8x16;
    #[link_name = "llvm.wasm.anytrue.v16i8"]
    fn llvm_any_true_i8x16(x: simd::i8x16) -> i32;

    #[link_name = "llvm.wasm.alltrue.v16i8"]
    fn llvm_i8x16_all_true(x: simd::i8x16) -> i32;
    #[link_name = "llvm.wasm.bitmask.v16i8"]
    fn llvm_bitmask_i8x16(a: simd::i8x16) -> i32;
    #[link_name = "llvm.wasm.narrow.signed.v16i8.v8i16"]
    fn llvm_narrow_i8x16_s(a: simd::i16x8, b: simd::i16x8) -> simd::i8x16;
    #[link_name = "llvm.wasm.narrow.unsigned.v16i8.v8i16"]
    fn llvm_narrow_i8x16_u(a: simd::i16x8, b: simd::i16x8) -> simd::i8x16;
    #[link_name = "llvm.wasm.avgr.unsigned.v16i8"]
    fn llvm_avgr_u_i8x16(a: simd::i8x16, b: simd::i8x16) -> simd::i8x16;

    #[link_name = "llvm.wasm.extadd.pairwise.signed.v8i16"]
    fn llvm_i16x8_extadd_pairwise_i8x16_s(x: simd::i8x16) -> simd::i16x8;
    #[link_name = "llvm.wasm.extadd.pairwise.unsigned.v8i16"]
    fn llvm_i16x8_extadd_pairwise_i8x16_u(x: simd::i8x16) -> simd::i16x8;
    #[link_name = "llvm.wasm.q15mulr.sat.signed"]
    fn llvm_q15mulr(a: simd::i16x8, b: simd::i16x8) -> simd::i16x8;
    #[link_name = "llvm.wasm.alltrue.v8i16"]
    fn llvm_i16x8_all_true(x: simd::i16x8) -> i32;
    #[link_name = "llvm.wasm.bitmask.v8i16"]
    fn llvm_bitmask_i16x8(a: simd::i16x8) -> i32;
    #[link_name = "llvm.wasm.narrow.signed.v8i16.v4i32"]
    fn llvm_narrow_i16x8_s(a: simd::i32x4, b: simd::i32x4) -> simd::i16x8;
    #[link_name = "llvm.wasm.narrow.unsigned.v8i16.v4i32"]
    fn llvm_narrow_i16x8_u(a: simd::i32x4, b: simd::i32x4) -> simd::i16x8;
    #[link_name = "llvm.wasm.avgr.unsigned.v8i16"]
    fn llvm_avgr_u_i16x8(a: simd::i16x8, b: simd::i16x8) -> simd::i16x8;

    #[link_name = "llvm.wasm.extadd.pairwise.signed.v4i32"]
    fn llvm_i32x4_extadd_pairwise_i16x8_s(x: simd::i16x8) -> simd::i32x4;
    #[link_name = "llvm.wasm.extadd.pairwise.unsigned.v4i32"]
    fn llvm_i32x4_extadd_pairwise_i16x8_u(x: simd::i16x8) -> simd::i32x4;
    #[link_name = "llvm.wasm.alltrue.v4i32"]
    fn llvm_i32x4_all_true(x: simd::i32x4) -> i32;
    #[link_name = "llvm.wasm.bitmask.v4i32"]
    fn llvm_bitmask_i32x4(a: simd::i32x4) -> i32;
    #[link_name = "llvm.wasm.dot"]
    fn llvm_i32x4_dot_i16x8_s(a: simd::i16x8, b: simd::i16x8) -> simd::i32x4;

    #[link_name = "llvm.wasm.alltrue.v2i64"]
    fn llvm_i64x2_all_true(x: simd::i64x2) -> i32;
    #[link_name = "llvm.wasm.bitmask.v2i64"]
    fn llvm_bitmask_i64x2(a: simd::i64x2) -> i32;

    #[link_name = "llvm.nearbyint.v4f32"]
    fn llvm_f32x4_nearest(x: simd::f32x4) -> simd::f32x4;
    #[link_name = "llvm.minimum.v4f32"]
    fn llvm_f32x4_min(x: simd::f32x4, y: simd::f32x4) -> simd::f32x4;
    #[link_name = "llvm.maximum.v4f32"]
    fn llvm_f32x4_max(x: simd::f32x4, y: simd::f32x4) -> simd::f32x4;

    #[link_name = "llvm.nearbyint.v2f64"]
    fn llvm_f64x2_nearest(x: simd::f64x2) -> simd::f64x2;
    #[link_name = "llvm.minimum.v2f64"]
    fn llvm_f64x2_min(x: simd::f64x2, y: simd::f64x2) -> simd::f64x2;
    #[link_name = "llvm.maximum.v2f64"]
    fn llvm_f64x2_max(x: simd::f64x2, y: simd::f64x2) -> simd::f64x2;
}

#[repr(C, packed)]
#[derive(Copy)]
struct Unaligned<T>(T);

impl<T: Copy> Clone for Unaligned<T> {
    fn clone(&self) -> Unaligned<T> {
        *self
    }
}

/// Loads a `v128` vector from the given heap address.
///
/// This intrinsic will emit a load with an alignment of 1. While this is
/// provided for completeness it is not strictly necessary, you can also load
/// the pointer directly:
///
/// ```rust,ignore
/// let a: &v128 = ...;
/// let value = unsafe { v128_load(a) };
/// // .. is the same as ..
/// let value = *a;
/// ```
///
/// The alignment of the load can be configured by doing a manual load without
/// this intrinsic.
///
/// # Unsafety
///
/// This intrinsic is unsafe because it takes a raw pointer as an argument, and
/// the pointer must be valid to load 16 bytes from. Note that there is no
/// alignment requirement on this pointer since this intrinsic performs a
/// 1-aligned load.
#[inline]
#[cfg_attr(test, assert_instr(v128.load))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.load"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub unsafe fn v128_load(m: *const v128) -> v128 {
    (*(m as *const Unaligned<v128>)).0
}

/// Load eight 8-bit integers and sign extend each one to a 16-bit lane
///
/// # Unsafety
///
/// This intrinsic is unsafe because it takes a raw pointer as an argument, and
/// the pointer must be valid to load 8 bytes from. Note that there is no
/// alignment requirement on this pointer since this intrinsic performs a
/// 1-aligned load.
#[inline]
#[cfg_attr(test, assert_instr(v128.load8x8_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.load8x8_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub unsafe fn i16x8_load_extend_i8x8(m: *const i8) -> v128 {
    let m = *(m as *const Unaligned<simd::i8x8>);
    simd_cast::<_, simd::i16x8>(m.0).v128()
}

/// Load eight 8-bit integers and zero extend each one to a 16-bit lane
///
/// # Unsafety
///
/// This intrinsic is unsafe because it takes a raw pointer as an argument, and
/// the pointer must be valid to load 8 bytes from. Note that there is no
/// alignment requirement on this pointer since this intrinsic performs a
/// 1-aligned load.
#[inline]
#[cfg_attr(test, assert_instr(v128.load8x8_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.load8x8_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub unsafe fn i16x8_load_extend_u8x8(m: *const u8) -> v128 {
    let m = *(m as *const Unaligned<simd::u8x8>);
    simd_cast::<_, simd::u16x8>(m.0).v128()
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i16x8_load_extend_u8x8 as u16x8_load_extend_u8x8;

/// Load four 16-bit integers and sign extend each one to a 32-bit lane
///
/// # Unsafety
///
/// This intrinsic is unsafe because it takes a raw pointer as an argument, and
/// the pointer must be valid to load 8 bytes from. Note that there is no
/// alignment requirement on this pointer since this intrinsic performs a
/// 1-aligned load.
#[inline]
#[cfg_attr(test, assert_instr(v128.load16x4_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.load16x4_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub unsafe fn i32x4_load_extend_i16x4(m: *const i16) -> v128 {
    let m = *(m as *const Unaligned<simd::i16x4>);
    simd_cast::<_, simd::i32x4>(m.0).v128()
}

/// Load four 16-bit integers and zero extend each one to a 32-bit lane
///
/// # Unsafety
///
/// This intrinsic is unsafe because it takes a raw pointer as an argument, and
/// the pointer must be valid to load 8 bytes from. Note that there is no
/// alignment requirement on this pointer since this intrinsic performs a
/// 1-aligned load.
#[inline]
#[cfg_attr(test, assert_instr(v128.load16x4_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.load16x4_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub unsafe fn i32x4_load_extend_u16x4(m: *const u16) -> v128 {
    let m = *(m as *const Unaligned<simd::u16x4>);
    simd_cast::<_, simd::u32x4>(m.0).v128()
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i32x4_load_extend_u16x4 as u32x4_load_extend_u16x4;

/// Load two 32-bit integers and sign extend each one to a 64-bit lane
///
/// # Unsafety
///
/// This intrinsic is unsafe because it takes a raw pointer as an argument, and
/// the pointer must be valid to load 8 bytes from. Note that there is no
/// alignment requirement on this pointer since this intrinsic performs a
/// 1-aligned load.
#[inline]
#[cfg_attr(test, assert_instr(v128.load32x2_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.load32x2_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub unsafe fn i64x2_load_extend_i32x2(m: *const i32) -> v128 {
    let m = *(m as *const Unaligned<simd::i32x2>);
    simd_cast::<_, simd::i64x2>(m.0).v128()
}

/// Load two 32-bit integers and zero extend each one to a 64-bit lane
///
/// # Unsafety
///
/// This intrinsic is unsafe because it takes a raw pointer as an argument, and
/// the pointer must be valid to load 8 bytes from. Note that there is no
/// alignment requirement on this pointer since this intrinsic performs a
/// 1-aligned load.
#[inline]
#[cfg_attr(test, assert_instr(v128.load32x2_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.load32x2_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub unsafe fn i64x2_load_extend_u32x2(m: *const u32) -> v128 {
    let m = *(m as *const Unaligned<simd::u32x2>);
    simd_cast::<_, simd::u64x2>(m.0).v128()
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i64x2_load_extend_u32x2 as u64x2_load_extend_u32x2;

/// Load a single element and splat to all lanes of a v128 vector.
///
/// While this intrinsic is provided for completeness it can also be replaced
/// with `u8x16_splat(*m)` and it should generate equivalent code (and also not
/// require `unsafe`).
///
/// # Unsafety
///
/// This intrinsic is unsafe because it takes a raw pointer as an argument, and
/// the pointer must be valid to load 1 byte from. Note that there is no
/// alignment requirement on this pointer since this intrinsic performs a
/// 1-aligned load.
#[inline]
#[cfg_attr(test, assert_instr(v128.load8_splat))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.load8_splat"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub unsafe fn v128_load8_splat(m: *const u8) -> v128 {
    u8x16_splat(*m)
}

/// Load a single element and splat to all lanes of a v128 vector.
///
/// While this intrinsic is provided for completeness it can also be replaced
/// with `u16x8_splat(*m)` and it should generate equivalent code (and also not
/// require `unsafe`).
///
/// # Unsafety
///
/// This intrinsic is unsafe because it takes a raw pointer as an argument, and
/// the pointer must be valid to load 2 bytes from. Note that there is no
/// alignment requirement on this pointer since this intrinsic performs a
/// 1-aligned load.
#[inline]
#[cfg_attr(test, assert_instr(v128.load16_splat))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.load16_splat"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub unsafe fn v128_load16_splat(m: *const u16) -> v128 {
    u16x8_splat(ptr::read_unaligned(m))
}

/// Load a single element and splat to all lanes of a v128 vector.
///
/// While this intrinsic is provided for completeness it can also be replaced
/// with `u32x4_splat(*m)` and it should generate equivalent code (and also not
/// require `unsafe`).
///
/// # Unsafety
///
/// This intrinsic is unsafe because it takes a raw pointer as an argument, and
/// the pointer must be valid to load 4 bytes from. Note that there is no
/// alignment requirement on this pointer since this intrinsic performs a
/// 1-aligned load.
#[inline]
#[cfg_attr(test, assert_instr(v128.load32_splat))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.load32_splat"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub unsafe fn v128_load32_splat(m: *const u32) -> v128 {
    u32x4_splat(ptr::read_unaligned(m))
}

/// Load a single element and splat to all lanes of a v128 vector.
///
/// While this intrinsic is provided for completeness it can also be replaced
/// with `u64x2_splat(*m)` and it should generate equivalent code (and also not
/// require `unsafe`).
///
/// # Unsafety
///
/// This intrinsic is unsafe because it takes a raw pointer as an argument, and
/// the pointer must be valid to load 8 bytes from. Note that there is no
/// alignment requirement on this pointer since this intrinsic performs a
/// 1-aligned load.
#[inline]
#[cfg_attr(test, assert_instr(v128.load64_splat))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.load64_splat"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub unsafe fn v128_load64_splat(m: *const u64) -> v128 {
    u64x2_splat(ptr::read_unaligned(m))
}

/// Load a 32-bit element into the low bits of the vector and sets all other
/// bits to zero.
///
/// This intrinsic is provided for completeness and is equivalent to `u32x4(*m,
/// 0, 0, 0)` (which doesn't require `unsafe`).
///
/// # Unsafety
///
/// This intrinsic is unsafe because it takes a raw pointer as an argument, and
/// the pointer must be valid to load 4 bytes from. Note that there is no
/// alignment requirement on this pointer since this intrinsic performs a
/// 1-aligned load.
#[inline]
#[cfg_attr(test, assert_instr(v128.load32_zero))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.load32_zero"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub unsafe fn v128_load32_zero(m: *const u32) -> v128 {
    u32x4(ptr::read_unaligned(m), 0, 0, 0)
}

/// Load a 64-bit element into the low bits of the vector and sets all other
/// bits to zero.
///
/// This intrinsic is provided for completeness and is equivalent to
/// `u64x2_replace_lane::<0>(u64x2(0, 0), *m)` (which doesn't require `unsafe`).
///
/// # Unsafety
///
/// This intrinsic is unsafe because it takes a raw pointer as an argument, and
/// the pointer must be valid to load 8 bytes from. Note that there is no
/// alignment requirement on this pointer since this intrinsic performs a
/// 1-aligned load.
#[inline]
#[cfg_attr(test, assert_instr(v128.load64_zero))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.load64_zero"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub unsafe fn v128_load64_zero(m: *const u64) -> v128 {
    u64x2_replace_lane::<0>(u64x2(0, 0), ptr::read_unaligned(m))
}

/// Stores a `v128` vector to the given heap address.
///
/// This intrinsic will emit a store with an alignment of 1. While this is
/// provided for completeness it is not strictly necessary, you can also store
/// the pointer directly:
///
/// ```rust,ignore
/// let a: &mut v128 = ...;
/// unsafe { v128_store(a, value) };
/// // .. is the same as ..
/// *a = value;
/// ```
///
/// The alignment of the store can be configured by doing a manual store without
/// this intrinsic.
///
/// # Unsafety
///
/// This intrinsic is unsafe because it takes a raw pointer as an argument, and
/// the pointer must be valid to store 16 bytes to. Note that there is no
/// alignment requirement on this pointer since this intrinsic performs a
/// 1-aligned store.
#[inline]
#[cfg_attr(test, assert_instr(v128.store))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.store"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub unsafe fn v128_store(m: *mut v128, a: v128) {
    *(m as *mut Unaligned<v128>) = Unaligned(a);
}

/// Loads an 8-bit value from `m` and sets lane `L` of `v` to that value.
///
/// This intrinsic is provided for completeness and is equivalent to
/// `u8x16_replace_lane::<L>(v, *m)` (which doesn't require `unsafe`).
///
/// # Unsafety
///
/// This intrinsic is unsafe because it takes a raw pointer as an argument, and
/// the pointer must be valid to load 1 byte from. Note that there is no
/// alignment requirement on this pointer since this intrinsic performs a
/// 1-aligned load.
#[inline]
#[cfg_attr(test, assert_instr(v128.load8_lane, L = 0))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.load8_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub unsafe fn v128_load8_lane<const L: usize>(v: v128, m: *const u8) -> v128 {
    u8x16_replace_lane::<L>(v, *m)
}

/// Loads a 16-bit value from `m` and sets lane `L` of `v` to that value.
///
/// This intrinsic is provided for completeness and is equivalent to
/// `u16x8_replace_lane::<L>(v, *m)` (which doesn't require `unsafe`).
///
/// # Unsafety
///
/// This intrinsic is unsafe because it takes a raw pointer as an argument, and
/// the pointer must be valid to load 2 bytes from. Note that there is no
/// alignment requirement on this pointer since this intrinsic performs a
/// 1-aligned load.
#[inline]
#[cfg_attr(test, assert_instr(v128.load16_lane, L = 0))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.load16_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub unsafe fn v128_load16_lane<const L: usize>(v: v128, m: *const u16) -> v128 {
    u16x8_replace_lane::<L>(v, ptr::read_unaligned(m))
}

/// Loads a 32-bit value from `m` and sets lane `L` of `v` to that value.
///
/// This intrinsic is provided for completeness and is equivalent to
/// `u32x4_replace_lane::<L>(v, *m)` (which doesn't require `unsafe`).
///
/// # Unsafety
///
/// This intrinsic is unsafe because it takes a raw pointer as an argument, and
/// the pointer must be valid to load 4 bytes from. Note that there is no
/// alignment requirement on this pointer since this intrinsic performs a
/// 1-aligned load.
#[inline]
#[cfg_attr(test, assert_instr(v128.load32_lane, L = 0))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.load32_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub unsafe fn v128_load32_lane<const L: usize>(v: v128, m: *const u32) -> v128 {
    u32x4_replace_lane::<L>(v, ptr::read_unaligned(m))
}

/// Loads a 64-bit value from `m` and sets lane `L` of `v` to that value.
///
/// This intrinsic is provided for completeness and is equivalent to
/// `u64x2_replace_lane::<L>(v, *m)` (which doesn't require `unsafe`).
///
/// # Unsafety
///
/// This intrinsic is unsafe because it takes a raw pointer as an argument, and
/// the pointer must be valid to load 8 bytes from. Note that there is no
/// alignment requirement on this pointer since this intrinsic performs a
/// 1-aligned load.
#[inline]
#[cfg_attr(test, assert_instr(v128.load64_lane, L = 0))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.load64_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub unsafe fn v128_load64_lane<const L: usize>(v: v128, m: *const u64) -> v128 {
    u64x2_replace_lane::<L>(v, ptr::read_unaligned(m))
}

/// Stores the 8-bit value from lane `L` of `v` into `m`
///
/// This intrinsic is provided for completeness and is equivalent to
/// `*m = u8x16_extract_lane::<L>(v)` (which doesn't require `unsafe`).
///
/// # Unsafety
///
/// This intrinsic is unsafe because it takes a raw pointer as an argument, and
/// the pointer must be valid to store 1 byte to. Note that there is no
/// alignment requirement on this pointer since this intrinsic performs a
/// 1-aligned store.
#[inline]
#[cfg_attr(test, assert_instr(v128.store8_lane, L = 0))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.store8_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub unsafe fn v128_store8_lane<const L: usize>(v: v128, m: *mut u8) {
    *m = u8x16_extract_lane::<L>(v);
}

/// Stores the 16-bit value from lane `L` of `v` into `m`
///
/// This intrinsic is provided for completeness and is equivalent to
/// `*m = u16x8_extract_lane::<L>(v)` (which doesn't require `unsafe`).
///
/// # Unsafety
///
/// This intrinsic is unsafe because it takes a raw pointer as an argument, and
/// the pointer must be valid to store 2 bytes to. Note that there is no
/// alignment requirement on this pointer since this intrinsic performs a
/// 1-aligned store.
#[inline]
#[cfg_attr(test, assert_instr(v128.store16_lane, L = 0))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.store16_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub unsafe fn v128_store16_lane<const L: usize>(v: v128, m: *mut u16) {
    ptr::write_unaligned(m, u16x8_extract_lane::<L>(v))
}

/// Stores the 32-bit value from lane `L` of `v` into `m`
///
/// This intrinsic is provided for completeness and is equivalent to
/// `*m = u32x4_extract_lane::<L>(v)` (which doesn't require `unsafe`).
///
/// # Unsafety
///
/// This intrinsic is unsafe because it takes a raw pointer as an argument, and
/// the pointer must be valid to store 4 bytes to. Note that there is no
/// alignment requirement on this pointer since this intrinsic performs a
/// 1-aligned store.
#[inline]
#[cfg_attr(test, assert_instr(v128.store32_lane, L = 0))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.store32_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub unsafe fn v128_store32_lane<const L: usize>(v: v128, m: *mut u32) {
    ptr::write_unaligned(m, u32x4_extract_lane::<L>(v))
}

/// Stores the 64-bit value from lane `L` of `v` into `m`
///
/// This intrinsic is provided for completeness and is equivalent to
/// `*m = u64x2_extract_lane::<L>(v)` (which doesn't require `unsafe`).
///
/// # Unsafety
///
/// This intrinsic is unsafe because it takes a raw pointer as an argument, and
/// the pointer must be valid to store 8 bytes to. Note that there is no
/// alignment requirement on this pointer since this intrinsic performs a
/// 1-aligned store.
#[inline]
#[cfg_attr(test, assert_instr(v128.store64_lane, L = 0))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.store64_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub unsafe fn v128_store64_lane<const L: usize>(v: v128, m: *mut u64) {
    ptr::write_unaligned(m, u64x2_extract_lane::<L>(v))
}

/// Materializes a SIMD value from the provided operands.
///
/// If possible this will generate a `v128.const` instruction, otherwise it may
/// be lowered to a sequence of instructions to materialize the vector value.
#[inline]
#[cfg_attr(
    test,
    assert_instr(
        v128.const,
        a0 = 0,
        a1 = 1,
        a2 = 2,
        a3 = 3,
        a4 = 4,
        a5 = 5,
        a6 = 6,
        a7 = 7,
        a8 = 8,
        a9 = 9,
        a10 = 10,
        a11 = 11,
        a12 = 12,
        a13 = 13,
        a14 = 14,
        a15 = 15,
    )
)]
#[doc(alias("v128.const"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
#[rustc_const_stable(feature = "wasm_simd", since = "1.54.0")]
#[target_feature(enable = "simd128")]
pub const fn i8x16(
    a0: i8,
    a1: i8,
    a2: i8,
    a3: i8,
    a4: i8,
    a5: i8,
    a6: i8,
    a7: i8,
    a8: i8,
    a9: i8,
    a10: i8,
    a11: i8,
    a12: i8,
    a13: i8,
    a14: i8,
    a15: i8,
) -> v128 {
    simd::i8x16::new(
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
    )
    .v128()
}

/// Materializes a SIMD value from the provided operands.
///
/// If possible this will generate a `v128.const` instruction, otherwise it may
/// be lowered to a sequence of instructions to materialize the vector value.
#[inline]
#[doc(alias("v128.const"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
#[rustc_const_stable(feature = "wasm_simd", since = "1.54.0")]
#[target_feature(enable = "simd128")]
pub const fn u8x16(
    a0: u8,
    a1: u8,
    a2: u8,
    a3: u8,
    a4: u8,
    a5: u8,
    a6: u8,
    a7: u8,
    a8: u8,
    a9: u8,
    a10: u8,
    a11: u8,
    a12: u8,
    a13: u8,
    a14: u8,
    a15: u8,
) -> v128 {
    simd::u8x16::new(
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
    )
    .v128()
}

/// Materializes a SIMD value from the provided operands.
///
/// If possible this will generate a `v128.const` instruction, otherwise it may
/// be lowered to a sequence of instructions to materialize the vector value.
#[inline]
#[cfg_attr(
    test,
    assert_instr(
        v128.const,
        a0 = 0,
        a1 = 1,
        a2 = 2,
        a3 = 3,
        a4 = 4,
        a5 = 5,
        a6 = 6,
        a7 = 7,
    )
)]
#[doc(alias("v128.const"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
#[rustc_const_stable(feature = "wasm_simd", since = "1.54.0")]
#[target_feature(enable = "simd128")]
pub const fn i16x8(a0: i16, a1: i16, a2: i16, a3: i16, a4: i16, a5: i16, a6: i16, a7: i16) -> v128 {
    simd::i16x8::new(a0, a1, a2, a3, a4, a5, a6, a7).v128()
}

/// Materializes a SIMD value from the provided operands.
///
/// If possible this will generate a `v128.const` instruction, otherwise it may
/// be lowered to a sequence of instructions to materialize the vector value.
#[inline]
#[doc(alias("v128.const"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
#[rustc_const_stable(feature = "wasm_simd", since = "1.54.0")]
#[target_feature(enable = "simd128")]
pub const fn u16x8(a0: u16, a1: u16, a2: u16, a3: u16, a4: u16, a5: u16, a6: u16, a7: u16) -> v128 {
    simd::u16x8::new(a0, a1, a2, a3, a4, a5, a6, a7).v128()
}

/// Materializes a SIMD value from the provided operands.
///
/// If possible this will generate a `v128.const` instruction, otherwise it may
/// be lowered to a sequence of instructions to materialize the vector value.
#[inline]
#[cfg_attr(test, assert_instr(v128.const, a0 = 0, a1 = 1, a2 = 2, a3 = 3))]
#[doc(alias("v128.const"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
#[rustc_const_stable(feature = "wasm_simd", since = "1.54.0")]
#[target_feature(enable = "simd128")]
pub const fn i32x4(a0: i32, a1: i32, a2: i32, a3: i32) -> v128 {
    simd::i32x4::new(a0, a1, a2, a3).v128()
}

/// Materializes a SIMD value from the provided operands.
///
/// If possible this will generate a `v128.const` instruction, otherwise it may
/// be lowered to a sequence of instructions to materialize the vector value.
#[inline]
#[doc(alias("v128.const"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
#[rustc_const_stable(feature = "wasm_simd", since = "1.54.0")]
#[target_feature(enable = "simd128")]
pub const fn u32x4(a0: u32, a1: u32, a2: u32, a3: u32) -> v128 {
    simd::u32x4::new(a0, a1, a2, a3).v128()
}

/// Materializes a SIMD value from the provided operands.
///
/// If possible this will generate a `v128.const` instruction, otherwise it may
/// be lowered to a sequence of instructions to materialize the vector value.
#[inline]
#[cfg_attr(test, assert_instr(v128.const, a0 = 1, a1 = 2))]
#[doc(alias("v128.const"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
#[rustc_const_stable(feature = "wasm_simd", since = "1.54.0")]
#[target_feature(enable = "simd128")]
pub const fn i64x2(a0: i64, a1: i64) -> v128 {
    simd::i64x2::new(a0, a1).v128()
}

/// Materializes a SIMD value from the provided operands.
///
/// If possible this will generate a `v128.const` instruction, otherwise it may
/// be lowered to a sequence of instructions to materialize the vector value.
#[inline]
#[doc(alias("v128.const"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
#[rustc_const_stable(feature = "wasm_simd", since = "1.54.0")]
#[target_feature(enable = "simd128")]
pub const fn u64x2(a0: u64, a1: u64) -> v128 {
    simd::u64x2::new(a0, a1).v128()
}

/// Materializes a SIMD value from the provided operands.
///
/// If possible this will generate a `v128.const` instruction, otherwise it may
/// be lowered to a sequence of instructions to materialize the vector value.
#[inline]
#[cfg_attr(test, assert_instr(v128.const, a0 = 0.0, a1 = 1.0, a2 = 2.0, a3 = 3.0))]
#[doc(alias("v128.const"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
#[rustc_const_stable(feature = "wasm_simd_const", since = "1.56.0")]
#[target_feature(enable = "simd128")]
pub const fn f32x4(a0: f32, a1: f32, a2: f32, a3: f32) -> v128 {
    simd::f32x4::new(a0, a1, a2, a3).v128()
}

/// Materializes a SIMD value from the provided operands.
///
/// If possible this will generate a `v128.const` instruction, otherwise it may
/// be lowered to a sequence of instructions to materialize the vector value.
#[inline]
#[cfg_attr(test, assert_instr(v128.const, a0 = 0.0, a1 = 1.0))]
#[doc(alias("v128.const"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
#[rustc_const_stable(feature = "wasm_simd_const", since = "1.56.0")]
#[target_feature(enable = "simd128")]
pub const fn f64x2(a0: f64, a1: f64) -> v128 {
    simd::f64x2::new(a0, a1).v128()
}

/// Returns a new vector with lanes selected from the lanes of the two input
/// vectors `$a` and `$b` specified in the 16 immediate operands.
///
/// The `$a` and `$b` expressions must have type `v128`, and this function
/// generates a wasm instruction that is encoded with 16 bytes providing the
/// indices of the elements to return. The indices `i` in range [0, 15] select
/// the `i`-th element of `a`. The indices in range [16, 31] select the `i -
/// 16`-th element of `b`.
///
/// Note that this is a macro due to the codegen requirements of all of the
/// index expressions `$i*` must be constant. A compiler error will be
/// generated if any of the expressions are not constant.
///
/// All indexes `$i*` must have the type `u32`.
#[inline]
#[cfg_attr(test,
    assert_instr(
        i8x16.shuffle,
        I0 = 0,
        I1 = 2,
        I2 = 4,
        I3 = 6,
        I4 = 8,
        I5 = 10,
        I6 = 12,
        I7 = 14,
        I8 = 16,
        I9 = 18,
        I10 = 20,
        I11 = 22,
        I12 = 24,
        I13 = 26,
        I14 = 28,
        I15 = 30,
    )
)]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.shuffle"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_shuffle<
    const I0: usize,
    const I1: usize,
    const I2: usize,
    const I3: usize,
    const I4: usize,
    const I5: usize,
    const I6: usize,
    const I7: usize,
    const I8: usize,
    const I9: usize,
    const I10: usize,
    const I11: usize,
    const I12: usize,
    const I13: usize,
    const I14: usize,
    const I15: usize,
>(
    a: v128,
    b: v128,
) -> v128 {
    static_assert!(I0 < 32);
    static_assert!(I1 < 32);
    static_assert!(I2 < 32);
    static_assert!(I3 < 32);
    static_assert!(I4 < 32);
    static_assert!(I5 < 32);
    static_assert!(I6 < 32);
    static_assert!(I7 < 32);
    static_assert!(I8 < 32);
    static_assert!(I9 < 32);
    static_assert!(I10 < 32);
    static_assert!(I11 < 32);
    static_assert!(I12 < 32);
    static_assert!(I13 < 32);
    static_assert!(I14 < 32);
    static_assert!(I15 < 32);
    let shuf: simd::u8x16 = unsafe {
        simd_shuffle!(
            a.as_u8x16(),
            b.as_u8x16(),
            [
                I0 as u32, I1 as u32, I2 as u32, I3 as u32, I4 as u32, I5 as u32, I6 as u32,
                I7 as u32, I8 as u32, I9 as u32, I10 as u32, I11 as u32, I12 as u32, I13 as u32,
                I14 as u32, I15 as u32,
            ],
        )
    };
    shuf.v128()
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i8x16_shuffle as u8x16_shuffle;

/// Same as [`i8x16_shuffle`], except operates as if the inputs were eight
/// 16-bit integers, only taking 8 indices to shuffle.
///
/// Indices in the range [0, 7] select from `a` while [8, 15] select from `b`.
/// Note that this will generate the `i8x16.shuffle` instruction, since there
/// is no native `i16x8.shuffle` instruction (there is no need for one since
/// `i8x16.shuffle` suffices).
#[inline]
#[cfg_attr(test,
    assert_instr(
        i8x16.shuffle,
        I0 = 0,
        I1 = 2,
        I2 = 4,
        I3 = 6,
        I4 = 8,
        I5 = 10,
        I6 = 12,
        I7 = 14,
    )
)]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.shuffle"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_shuffle<
    const I0: usize,
    const I1: usize,
    const I2: usize,
    const I3: usize,
    const I4: usize,
    const I5: usize,
    const I6: usize,
    const I7: usize,
>(
    a: v128,
    b: v128,
) -> v128 {
    static_assert!(I0 < 16);
    static_assert!(I1 < 16);
    static_assert!(I2 < 16);
    static_assert!(I3 < 16);
    static_assert!(I4 < 16);
    static_assert!(I5 < 16);
    static_assert!(I6 < 16);
    static_assert!(I7 < 16);
    let shuf: simd::u16x8 = unsafe {
        simd_shuffle!(
            a.as_u16x8(),
            b.as_u16x8(),
            [
                I0 as u32, I1 as u32, I2 as u32, I3 as u32, I4 as u32, I5 as u32, I6 as u32,
                I7 as u32,
            ],
        )
    };
    shuf.v128()
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i16x8_shuffle as u16x8_shuffle;

/// Same as [`i8x16_shuffle`], except operates as if the inputs were four
/// 32-bit integers, only taking 4 indices to shuffle.
///
/// Indices in the range [0, 3] select from `a` while [4, 7] select from `b`.
/// Note that this will generate the `i8x16.shuffle` instruction, since there
/// is no native `i32x4.shuffle` instruction (there is no need for one since
/// `i8x16.shuffle` suffices).
#[inline]
#[cfg_attr(test, assert_instr(i8x16.shuffle, I0 = 0, I1 = 2, I2 = 4, I3 = 6))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.shuffle"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_shuffle<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
    a: v128,
    b: v128,
) -> v128 {
    static_assert!(I0 < 8);
    static_assert!(I1 < 8);
    static_assert!(I2 < 8);
    static_assert!(I3 < 8);
    let shuf: simd::u32x4 = unsafe {
        simd_shuffle!(
            a.as_u32x4(),
            b.as_u32x4(),
            [I0 as u32, I1 as u32, I2 as u32, I3 as u32],
        )
    };
    shuf.v128()
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i32x4_shuffle as u32x4_shuffle;

/// Same as [`i8x16_shuffle`], except operates as if the inputs were two
/// 64-bit integers, only taking 2 indices to shuffle.
///
/// Indices in the range [0, 1] select from `a` while [2, 3] select from `b`.
/// Note that this will generate the `v8x16.shuffle` instruction, since there
/// is no native `i64x2.shuffle` instruction (there is no need for one since
/// `i8x16.shuffle` suffices).
#[inline]
#[cfg_attr(test, assert_instr(i8x16.shuffle, I0 = 0, I1 = 2))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.shuffle"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_shuffle<const I0: usize, const I1: usize>(a: v128, b: v128) -> v128 {
    static_assert!(I0 < 4);
    static_assert!(I1 < 4);
    let shuf: simd::u64x2 =
        unsafe { simd_shuffle!(a.as_u64x2(), b.as_u64x2(), [I0 as u32, I1 as u32]) };
    shuf.v128()
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i64x2_shuffle as u64x2_shuffle;

/// Extracts a lane from a 128-bit vector interpreted as 16 packed i8 numbers.
///
/// Extracts the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.extract_lane_s, N = 3))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.extract_lane_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_extract_lane<const N: usize>(a: v128) -> i8 {
    static_assert!(N < 16);
    unsafe { simd_extract!(a.as_i8x16(), N as u32) }
}

/// Extracts a lane from a 128-bit vector interpreted as 16 packed u8 numbers.
///
/// Extracts the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.extract_lane_u, N = 3))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.extract_lane_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u8x16_extract_lane<const N: usize>(a: v128) -> u8 {
    static_assert!(N < 16);
    unsafe { simd_extract!(a.as_u8x16(), N as u32) }
}

/// Replaces a lane from a 128-bit vector interpreted as 16 packed i8 numbers.
///
/// Replaces the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.replace_lane, N = 2))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.replace_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_replace_lane<const N: usize>(a: v128, val: i8) -> v128 {
    static_assert!(N < 16);
    unsafe { simd_insert!(a.as_i8x16(), N as u32, val).v128() }
}

/// Replaces a lane from a 128-bit vector interpreted as 16 packed u8 numbers.
///
/// Replaces the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.replace_lane, N = 2))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.replace_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u8x16_replace_lane<const N: usize>(a: v128, val: u8) -> v128 {
    static_assert!(N < 16);
    unsafe { simd_insert!(a.as_u8x16(), N as u32, val).v128() }
}

/// Extracts a lane from a 128-bit vector interpreted as 8 packed i16 numbers.
///
/// Extracts a the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.extract_lane_s, N = 2))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.extract_lane_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_extract_lane<const N: usize>(a: v128) -> i16 {
    static_assert!(N < 8);
    unsafe { simd_extract!(a.as_i16x8(), N as u32) }
}

/// Extracts a lane from a 128-bit vector interpreted as 8 packed u16 numbers.
///
/// Extracts a the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.extract_lane_u, N = 2))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.extract_lane_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u16x8_extract_lane<const N: usize>(a: v128) -> u16 {
    static_assert!(N < 8);
    unsafe { simd_extract!(a.as_u16x8(), N as u32) }
}

/// Replaces a lane from a 128-bit vector interpreted as 8 packed i16 numbers.
///
/// Replaces the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.replace_lane, N = 2))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.replace_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_replace_lane<const N: usize>(a: v128, val: i16) -> v128 {
    static_assert!(N < 8);
    unsafe { simd_insert!(a.as_i16x8(), N as u32, val).v128() }
}

/// Replaces a lane from a 128-bit vector interpreted as 8 packed u16 numbers.
///
/// Replaces the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.replace_lane, N = 2))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.replace_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u16x8_replace_lane<const N: usize>(a: v128, val: u16) -> v128 {
    static_assert!(N < 8);
    unsafe { simd_insert!(a.as_u16x8(), N as u32, val).v128() }
}

/// Extracts a lane from a 128-bit vector interpreted as 4 packed i32 numbers.
///
/// Extracts the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.extract_lane, N = 2))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.extract_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_extract_lane<const N: usize>(a: v128) -> i32 {
    static_assert!(N < 4);
    unsafe { simd_extract!(a.as_i32x4(), N as u32) }
}

/// Extracts a lane from a 128-bit vector interpreted as 4 packed u32 numbers.
///
/// Extracts the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.extract_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u32x4_extract_lane<const N: usize>(a: v128) -> u32 {
    i32x4_extract_lane::<N>(a) as u32
}

/// Replaces a lane from a 128-bit vector interpreted as 4 packed i32 numbers.
///
/// Replaces the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.replace_lane, N = 2))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.replace_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_replace_lane<const N: usize>(a: v128, val: i32) -> v128 {
    static_assert!(N < 4);
    unsafe { simd_insert!(a.as_i32x4(), N as u32, val).v128() }
}

/// Replaces a lane from a 128-bit vector interpreted as 4 packed u32 numbers.
///
/// Replaces the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.replace_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u32x4_replace_lane<const N: usize>(a: v128, val: u32) -> v128 {
    i32x4_replace_lane::<N>(a, val as i32)
}

/// Extracts a lane from a 128-bit vector interpreted as 2 packed i64 numbers.
///
/// Extracts the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.extract_lane, N = 1))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.extract_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_extract_lane<const N: usize>(a: v128) -> i64 {
    static_assert!(N < 2);
    unsafe { simd_extract!(a.as_i64x2(), N as u32) }
}

/// Extracts a lane from a 128-bit vector interpreted as 2 packed u64 numbers.
///
/// Extracts the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.extract_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u64x2_extract_lane<const N: usize>(a: v128) -> u64 {
    i64x2_extract_lane::<N>(a) as u64
}

/// Replaces a lane from a 128-bit vector interpreted as 2 packed i64 numbers.
///
/// Replaces the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.replace_lane, N = 0))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.replace_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_replace_lane<const N: usize>(a: v128, val: i64) -> v128 {
    static_assert!(N < 2);
    unsafe { simd_insert!(a.as_i64x2(), N as u32, val).v128() }
}

/// Replaces a lane from a 128-bit vector interpreted as 2 packed u64 numbers.
///
/// Replaces the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.replace_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u64x2_replace_lane<const N: usize>(a: v128, val: u64) -> v128 {
    i64x2_replace_lane::<N>(a, val as i64)
}

/// Extracts a lane from a 128-bit vector interpreted as 4 packed f32 numbers.
///
/// Extracts the scalar value of lane specified fn the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.extract_lane, N = 1))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.extract_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_extract_lane<const N: usize>(a: v128) -> f32 {
    static_assert!(N < 4);
    unsafe { simd_extract!(a.as_f32x4(), N as u32) }
}

/// Replaces a lane from a 128-bit vector interpreted as 4 packed f32 numbers.
///
/// Replaces the scalar value of lane specified fn the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.replace_lane, N = 1))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.replace_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_replace_lane<const N: usize>(a: v128, val: f32) -> v128 {
    static_assert!(N < 4);
    unsafe { simd_insert!(a.as_f32x4(), N as u32, val).v128() }
}

/// Extracts a lane from a 128-bit vector interpreted as 2 packed f64 numbers.
///
/// Extracts the scalar value of lane specified fn the immediate mode operand
/// `N` from `a`. If `N` fs out of bounds then it is a compile time error.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.extract_lane, N = 1))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.extract_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_extract_lane<const N: usize>(a: v128) -> f64 {
    static_assert!(N < 2);
    unsafe { simd_extract!(a.as_f64x2(), N as u32) }
}

/// Replaces a lane from a 128-bit vector interpreted as 2 packed f64 numbers.
///
/// Replaces the scalar value of lane specified in the immediate mode operand
/// `N` from `a`. If `N` is out of bounds then it is a compile time error.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.replace_lane, N = 1))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.replace_lane"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_replace_lane<const N: usize>(a: v128, val: f64) -> v128 {
    static_assert!(N < 2);
    unsafe { simd_insert!(a.as_f64x2(), N as u32, val).v128() }
}

/// Returns a new vector with lanes selected from the lanes of the first input
/// vector `a` specified in the second input vector `s`.
///
/// The indices `i` in range [0, 15] select the `i`-th element of `a`. For
/// indices outside of the range the resulting lane is 0.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.swizzle))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.swizzle"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_swizzle(a: v128, s: v128) -> v128 {
    unsafe { llvm_swizzle(a.as_i8x16(), s.as_i8x16()).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i8x16_swizzle as u8x16_swizzle;

/// Creates a vector with identical lanes.
///
/// Constructs a vector with `x` replicated to all 16 lanes.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.splat))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.splat"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_splat(a: i8) -> v128 {
    simd::i8x16::splat(a).v128()
}

/// Creates a vector with identical lanes.
///
/// Constructs a vector with `x` replicated to all 16 lanes.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.splat))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.splat"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u8x16_splat(a: u8) -> v128 {
    simd::u8x16::splat(a).v128()
}

/// Creates a vector with identical lanes.
///
/// Construct a vector with `x` replicated to all 8 lanes.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.splat))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.splat"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_splat(a: i16) -> v128 {
    simd::i16x8::splat(a).v128()
}

/// Creates a vector with identical lanes.
///
/// Construct a vector with `x` replicated to all 8 lanes.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.splat))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.splat"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u16x8_splat(a: u16) -> v128 {
    simd::u16x8::splat(a).v128()
}

/// Creates a vector with identical lanes.
///
/// Constructs a vector with `x` replicated to all 4 lanes.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.splat))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.splat"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_splat(a: i32) -> v128 {
    simd::i32x4::splat(a).v128()
}

/// Creates a vector with identical lanes.
///
/// Constructs a vector with `x` replicated to all 4 lanes.
#[inline]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.splat"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u32x4_splat(a: u32) -> v128 {
    i32x4_splat(a as i32)
}

/// Creates a vector with identical lanes.
///
/// Construct a vector with `x` replicated to all 2 lanes.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.splat))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.splat"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_splat(a: i64) -> v128 {
    simd::i64x2::splat(a).v128()
}

/// Creates a vector with identical lanes.
///
/// Construct a vector with `x` replicated to all 2 lanes.
#[inline]
#[target_feature(enable = "simd128")]
#[doc(alias("u64x2.splat"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u64x2_splat(a: u64) -> v128 {
    i64x2_splat(a as i64)
}

/// Creates a vector with identical lanes.
///
/// Constructs a vector with `x` replicated to all 4 lanes.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.splat))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.splat"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_splat(a: f32) -> v128 {
    simd::f32x4::splat(a).v128()
}

/// Creates a vector with identical lanes.
///
/// Constructs a vector with `x` replicated to all 2 lanes.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.splat))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.splat"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_splat(a: f64) -> v128 {
    simd::f64x2::splat(a).v128()
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// integers.
///
/// Returns a new vector where each lane is all ones if the corresponding input elements
/// were equal, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.eq))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.eq"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_eq(a: v128, b: v128) -> v128 {
    unsafe { simd_eq::<_, simd::i8x16>(a.as_i8x16(), b.as_i8x16()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// integers.
///
/// Returns a new vector where each lane is all ones if the corresponding input elements
/// were not equal, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.ne))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.ne"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_ne(a: v128, b: v128) -> v128 {
    unsafe { simd_ne::<_, simd::i8x16>(a.as_i8x16(), b.as_i8x16()).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i8x16_eq as u8x16_eq;
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i8x16_ne as u8x16_ne;

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is less than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.lt_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.lt_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_lt(a: v128, b: v128) -> v128 {
    unsafe { simd_lt::<_, simd::i8x16>(a.as_i8x16(), b.as_i8x16()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is less than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.lt_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.lt_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u8x16_lt(a: v128, b: v128) -> v128 {
    unsafe { simd_lt::<_, simd::i8x16>(a.as_u8x16(), b.as_u8x16()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is greater than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.gt_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.gt_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_gt(a: v128, b: v128) -> v128 {
    unsafe { simd_gt::<_, simd::i8x16>(a.as_i8x16(), b.as_i8x16()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is greater than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.gt_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.gt_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u8x16_gt(a: v128, b: v128) -> v128 {
    unsafe { simd_gt::<_, simd::i8x16>(a.as_u8x16(), b.as_u8x16()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is less than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.le_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.le_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_le(a: v128, b: v128) -> v128 {
    unsafe { simd_le::<_, simd::i8x16>(a.as_i8x16(), b.as_i8x16()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is less than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.le_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.le_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u8x16_le(a: v128, b: v128) -> v128 {
    unsafe { simd_le::<_, simd::i8x16>(a.as_u8x16(), b.as_u8x16()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is greater than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.ge_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.ge_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_ge(a: v128, b: v128) -> v128 {
    unsafe { simd_ge::<_, simd::i8x16>(a.as_i8x16(), b.as_i8x16()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 16 eight-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is greater than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.ge_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.ge_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u8x16_ge(a: v128, b: v128) -> v128 {
    unsafe { simd_ge::<_, simd::i8x16>(a.as_u8x16(), b.as_u8x16()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// integers.
///
/// Returns a new vector where each lane is all ones if the corresponding input elements
/// were equal, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.eq))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.eq"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_eq(a: v128, b: v128) -> v128 {
    unsafe { simd_eq::<_, simd::i16x8>(a.as_i16x8(), b.as_i16x8()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// integers.
///
/// Returns a new vector where each lane is all ones if the corresponding input elements
/// were not equal, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.ne))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.ne"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_ne(a: v128, b: v128) -> v128 {
    unsafe { simd_ne::<_, simd::i16x8>(a.as_i16x8(), b.as_i16x8()).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i16x8_eq as u16x8_eq;
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i16x8_ne as u16x8_ne;

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is less than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.lt_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.lt_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_lt(a: v128, b: v128) -> v128 {
    unsafe { simd_lt::<_, simd::i16x8>(a.as_i16x8(), b.as_i16x8()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is less than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.lt_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.lt_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u16x8_lt(a: v128, b: v128) -> v128 {
    unsafe { simd_lt::<_, simd::i16x8>(a.as_u16x8(), b.as_u16x8()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is greater than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.gt_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.gt_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_gt(a: v128, b: v128) -> v128 {
    unsafe { simd_gt::<_, simd::i16x8>(a.as_i16x8(), b.as_i16x8()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is greater than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.gt_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.gt_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u16x8_gt(a: v128, b: v128) -> v128 {
    unsafe { simd_gt::<_, simd::i16x8>(a.as_u16x8(), b.as_u16x8()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is less than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.le_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.le_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_le(a: v128, b: v128) -> v128 {
    unsafe { simd_le::<_, simd::i16x8>(a.as_i16x8(), b.as_i16x8()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is less than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.le_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.le_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u16x8_le(a: v128, b: v128) -> v128 {
    unsafe { simd_le::<_, simd::i16x8>(a.as_u16x8(), b.as_u16x8()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is greater than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.ge_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.ge_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_ge(a: v128, b: v128) -> v128 {
    unsafe { simd_ge::<_, simd::i16x8>(a.as_i16x8(), b.as_i16x8()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 8 sixteen-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is greater than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.ge_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.ge_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u16x8_ge(a: v128, b: v128) -> v128 {
    unsafe { simd_ge::<_, simd::i16x8>(a.as_u16x8(), b.as_u16x8()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// integers.
///
/// Returns a new vector where each lane is all ones if the corresponding input elements
/// were equal, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.eq))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.eq"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_eq(a: v128, b: v128) -> v128 {
    unsafe { simd_eq::<_, simd::i32x4>(a.as_i32x4(), b.as_i32x4()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// integers.
///
/// Returns a new vector where each lane is all ones if the corresponding input elements
/// were not equal, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.ne))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.ne"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_ne(a: v128, b: v128) -> v128 {
    unsafe { simd_ne::<_, simd::i32x4>(a.as_i32x4(), b.as_i32x4()).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i32x4_eq as u32x4_eq;
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i32x4_ne as u32x4_ne;

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is less than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.lt_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.lt_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_lt(a: v128, b: v128) -> v128 {
    unsafe { simd_lt::<_, simd::i32x4>(a.as_i32x4(), b.as_i32x4()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is less than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.lt_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.lt_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u32x4_lt(a: v128, b: v128) -> v128 {
    unsafe { simd_lt::<_, simd::i32x4>(a.as_u32x4(), b.as_u32x4()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is greater than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.gt_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.gt_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_gt(a: v128, b: v128) -> v128 {
    unsafe { simd_gt::<_, simd::i32x4>(a.as_i32x4(), b.as_i32x4()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is greater than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.gt_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.gt_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u32x4_gt(a: v128, b: v128) -> v128 {
    unsafe { simd_gt::<_, simd::i32x4>(a.as_u32x4(), b.as_u32x4()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is less than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.le_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.le_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_le(a: v128, b: v128) -> v128 {
    unsafe { simd_le::<_, simd::i32x4>(a.as_i32x4(), b.as_i32x4()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is less than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.le_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.le_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u32x4_le(a: v128, b: v128) -> v128 {
    unsafe { simd_le::<_, simd::i32x4>(a.as_u32x4(), b.as_u32x4()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is greater than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.ge_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.ge_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_ge(a: v128, b: v128) -> v128 {
    unsafe { simd_ge::<_, simd::i32x4>(a.as_i32x4(), b.as_i32x4()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// unsigned integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is greater than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.ge_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.ge_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u32x4_ge(a: v128, b: v128) -> v128 {
    unsafe { simd_ge::<_, simd::i32x4>(a.as_u32x4(), b.as_u32x4()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// integers.
///
/// Returns a new vector where each lane is all ones if the corresponding input elements
/// were equal, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.eq))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.eq"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_eq(a: v128, b: v128) -> v128 {
    unsafe { simd_eq::<_, simd::i64x2>(a.as_i64x2(), b.as_i64x2()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// integers.
///
/// Returns a new vector where each lane is all ones if the corresponding input elements
/// were not equal, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.ne))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.ne"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_ne(a: v128, b: v128) -> v128 {
    unsafe { simd_ne::<_, simd::i64x2>(a.as_i64x2(), b.as_i64x2()).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i64x2_eq as u64x2_eq;
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i64x2_ne as u64x2_ne;

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is less than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.lt_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.lt_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_lt(a: v128, b: v128) -> v128 {
    unsafe { simd_lt::<_, simd::i64x2>(a.as_i64x2(), b.as_i64x2()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is greater than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.gt_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.gt_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_gt(a: v128, b: v128) -> v128 {
    unsafe { simd_gt::<_, simd::i64x2>(a.as_i64x2(), b.as_i64x2()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is less than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.le_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.le_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_le(a: v128, b: v128) -> v128 {
    unsafe { simd_le::<_, simd::i64x2>(a.as_i64x2(), b.as_i64x2()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// signed integers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is greater than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.ge_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.ge_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_ge(a: v128, b: v128) -> v128 {
    unsafe { simd_ge::<_, simd::i64x2>(a.as_i64x2(), b.as_i64x2()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the corresponding input elements
/// were equal, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.eq))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.eq"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_eq(a: v128, b: v128) -> v128 {
    unsafe { simd_eq::<_, simd::i32x4>(a.as_f32x4(), b.as_f32x4()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the corresponding input elements
/// were not equal, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.ne))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.ne"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_ne(a: v128, b: v128) -> v128 {
    unsafe { simd_ne::<_, simd::i32x4>(a.as_f32x4(), b.as_f32x4()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is less than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.lt))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.lt"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_lt(a: v128, b: v128) -> v128 {
    unsafe { simd_lt::<_, simd::i32x4>(a.as_f32x4(), b.as_f32x4()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is greater than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.gt))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.gt"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_gt(a: v128, b: v128) -> v128 {
    unsafe { simd_gt::<_, simd::i32x4>(a.as_f32x4(), b.as_f32x4()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is less than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.le))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.le"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_le(a: v128, b: v128) -> v128 {
    unsafe { simd_le::<_, simd::i32x4>(a.as_f32x4(), b.as_f32x4()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 4 thirty-two-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is greater than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.ge))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.ge"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_ge(a: v128, b: v128) -> v128 {
    unsafe { simd_ge::<_, simd::i32x4>(a.as_f32x4(), b.as_f32x4()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the corresponding input elements
/// were equal, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.eq))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.eq"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_eq(a: v128, b: v128) -> v128 {
    unsafe { simd_eq::<_, simd::i64x2>(a.as_f64x2(), b.as_f64x2()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the corresponding input elements
/// were not equal, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.ne))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.ne"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_ne(a: v128, b: v128) -> v128 {
    unsafe { simd_ne::<_, simd::i64x2>(a.as_f64x2(), b.as_f64x2()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is less than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.lt))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.lt"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_lt(a: v128, b: v128) -> v128 {
    unsafe { simd_lt::<_, simd::i64x2>(a.as_f64x2(), b.as_f64x2()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is greater than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.gt))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.gt"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_gt(a: v128, b: v128) -> v128 {
    unsafe { simd_gt::<_, simd::i64x2>(a.as_f64x2(), b.as_f64x2()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is less than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.le))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.le"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_le(a: v128, b: v128) -> v128 {
    unsafe { simd_le::<_, simd::i64x2>(a.as_f64x2(), b.as_f64x2()).v128() }
}

/// Compares two 128-bit vectors as if they were two vectors of 2 sixty-four-bit
/// floating point numbers.
///
/// Returns a new vector where each lane is all ones if the lane-wise left
/// element is greater than the right element, or all zeros otherwise.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.ge))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.ge"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_ge(a: v128, b: v128) -> v128 {
    unsafe { simd_ge::<_, simd::i64x2>(a.as_f64x2(), b.as_f64x2()).v128() }
}

/// Flips each bit of the 128-bit input vector.
#[inline]
#[cfg_attr(test, assert_instr(v128.not))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.not"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn v128_not(a: v128) -> v128 {
    unsafe { simd_xor(a.as_i64x2(), simd::i64x2::new(!0, !0)).v128() }
}

/// Performs a bitwise and of the two input 128-bit vectors, returning the
/// resulting vector.
#[inline]
#[cfg_attr(test, assert_instr(v128.and))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.and"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn v128_and(a: v128, b: v128) -> v128 {
    unsafe { simd_and(a.as_i64x2(), b.as_i64x2()).v128() }
}

/// Bitwise AND of bits of `a` and the logical inverse of bits of `b`.
///
/// This operation is equivalent to `v128.and(a, v128.not(b))`
#[inline]
#[cfg_attr(test, assert_instr(v128.andnot))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.andnot"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn v128_andnot(a: v128, b: v128) -> v128 {
    unsafe {
        simd_and(
            a.as_i64x2(),
            simd_xor(b.as_i64x2(), simd::i64x2::new(-1, -1)),
        )
        .v128()
    }
}

/// Performs a bitwise or of the two input 128-bit vectors, returning the
/// resulting vector.
#[inline]
#[cfg_attr(test, assert_instr(v128.or))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.or"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn v128_or(a: v128, b: v128) -> v128 {
    unsafe { simd_or(a.as_i64x2(), b.as_i64x2()).v128() }
}

/// Performs a bitwise xor of the two input 128-bit vectors, returning the
/// resulting vector.
#[inline]
#[cfg_attr(test, assert_instr(v128.xor))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.xor"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn v128_xor(a: v128, b: v128) -> v128 {
    unsafe { simd_xor(a.as_i64x2(), b.as_i64x2()).v128() }
}

/// Use the bitmask in `c` to select bits from `v1` when 1 and `v2` when 0.
#[inline]
#[cfg_attr(test, assert_instr(v128.bitselect))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.bitselect"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn v128_bitselect(v1: v128, v2: v128, c: v128) -> v128 {
    unsafe { llvm_bitselect(v1.as_i8x16(), v2.as_i8x16(), c.as_i8x16()).v128() }
}

/// Returns `true` if any bit in `a` is set, or `false` otherwise.
#[inline]
#[cfg_attr(test, assert_instr(v128.any_true))]
#[target_feature(enable = "simd128")]
#[doc(alias("v128.any_true"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn v128_any_true(a: v128) -> bool {
    unsafe { llvm_any_true_i8x16(a.as_i8x16()) != 0 }
}

/// Lane-wise wrapping absolute value.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.abs))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.abs"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_abs(a: v128) -> v128 {
    unsafe {
        let a = a.as_i8x16();
        let zero = simd::i8x16::ZERO;
        simd_select::<simd::m8x16, simd::i8x16>(simd_lt(a, zero), simd_sub(zero, a), a).v128()
    }
}

/// Negates a 128-bit vectors interpreted as sixteen 8-bit signed integers
#[inline]
#[cfg_attr(test, assert_instr(i8x16.neg))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.neg"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_neg(a: v128) -> v128 {
    unsafe { simd_mul(a.as_i8x16(), simd::i8x16::splat(-1)).v128() }
}

/// Count the number of bits set to one within each lane.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.popcnt))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.popcnt"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_popcnt(v: v128) -> v128 {
    unsafe { simd_ctpop(v.as_i8x16()).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i8x16_popcnt as u8x16_popcnt;

/// Returns true if all lanes are non-zero, false otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.all_true))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.all_true"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_all_true(a: v128) -> bool {
    unsafe { llvm_i8x16_all_true(a.as_i8x16()) != 0 }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i8x16_all_true as u8x16_all_true;

/// Extracts the high bit for each lane in `a` and produce a scalar mask with
/// all bits concatenated.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.bitmask))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.bitmask"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_bitmask(a: v128) -> u16 {
    unsafe { llvm_bitmask_i8x16(a.as_i8x16()) as u16 }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i8x16_bitmask as u8x16_bitmask;

/// Converts two input vectors into a smaller lane vector by narrowing each
/// lane.
///
/// Signed saturation to 0x7f or 0x80 is used and the input lanes are always
/// interpreted as signed integers.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.narrow_i16x8_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.narrow_i16x8_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_narrow_i16x8(a: v128, b: v128) -> v128 {
    unsafe { llvm_narrow_i8x16_s(a.as_i16x8(), b.as_i16x8()).v128() }
}

/// Converts two input vectors into a smaller lane vector by narrowing each
/// lane.
///
/// Signed saturation to 0x00 or 0xff is used and the input lanes are always
/// interpreted as signed integers.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.narrow_i16x8_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.narrow_i16x8_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u8x16_narrow_i16x8(a: v128, b: v128) -> v128 {
    unsafe { llvm_narrow_i8x16_u(a.as_i16x8(), b.as_i16x8()).v128() }
}

/// Shifts each lane to the left by the specified number of bits.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.shl))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.shl"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_shl(a: v128, amt: u32) -> v128 {
    // SAFETY: the safety of this intrinsic relies on the fact that the
    // shift amount for each lane is less than the number of bits in the input
    // lane. In this case the input has 8-bit lanes but the shift amount above
    // is `u32`, so a mask is required to discard all the upper bits of `amt` to
    // ensure that the safety condition is met.
    //
    // Note that this is distinct from the behavior of the native WebAssembly
    // instruction here where WebAssembly defines this instruction as performing
    // a mask as well. This is nonetheless required since this must have defined
    // semantics in LLVM, not just WebAssembly.
    //
    // Finally note that this mask operation is not actually emitted into the
    // final binary itself. LLVM understands that the wasm operation implicitly
    // masks, so it knows this mask operation is redundant.
    //
    // Basically the extra mask here is required as a bridge from the documented
    // semantics through LLVM back out to WebAssembly. Both ends have the
    // documented semantics, and the mask is required by LLVM in the middle.
    unsafe { simd_shl(a.as_i8x16(), simd::i8x16::splat((amt & 0x7) as i8)).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i8x16_shl as u8x16_shl;

/// Shifts each lane to the right by the specified number of bits, sign
/// extending.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.shr_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.shr_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_shr(a: v128, amt: u32) -> v128 {
    // SAFETY: see i8x16_shl for more documentation why this is unsafe,
    // essentially the shift amount must be valid hence the mask.
    unsafe { simd_shr(a.as_i8x16(), simd::i8x16::splat((amt & 0x7) as i8)).v128() }
}

/// Shifts each lane to the right by the specified number of bits, shifting in
/// zeros.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.shr_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.shr_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u8x16_shr(a: v128, amt: u32) -> v128 {
    // SAFETY: see i8x16_shl for more documentation why this is unsafe,
    // essentially the shift amount must be valid hence the mask.
    unsafe { simd_shr(a.as_u8x16(), simd::u8x16::splat((amt & 0x7) as u8)).v128() }
}

/// Adds two 128-bit vectors as if they were two packed sixteen 8-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.add))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.add"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_add(a: v128, b: v128) -> v128 {
    unsafe { simd_add(a.as_i8x16(), b.as_i8x16()).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i8x16_add as u8x16_add;

/// Adds two 128-bit vectors as if they were two packed sixteen 8-bit signed
/// integers, saturating on overflow to `i8::MAX`.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.add_sat_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.add_sat_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_add_sat(a: v128, b: v128) -> v128 {
    unsafe { simd_saturating_add(a.as_i8x16(), b.as_i8x16()).v128() }
}

/// Adds two 128-bit vectors as if they were two packed sixteen 8-bit unsigned
/// integers, saturating on overflow to `u8::MAX`.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.add_sat_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.add_sat_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u8x16_add_sat(a: v128, b: v128) -> v128 {
    unsafe { simd_saturating_add(a.as_u8x16(), b.as_u8x16()).v128() }
}

/// Subtracts two 128-bit vectors as if they were two packed sixteen 8-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.sub))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.sub"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_sub(a: v128, b: v128) -> v128 {
    unsafe { simd_sub(a.as_i8x16(), b.as_i8x16()).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i8x16_sub as u8x16_sub;

/// Subtracts two 128-bit vectors as if they were two packed sixteen 8-bit
/// signed integers, saturating on overflow to `i8::MIN`.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.sub_sat_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.sub_sat_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_sub_sat(a: v128, b: v128) -> v128 {
    unsafe { simd_saturating_sub(a.as_i8x16(), b.as_i8x16()).v128() }
}

/// Subtracts two 128-bit vectors as if they were two packed sixteen 8-bit
/// unsigned integers, saturating on overflow to 0.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.sub_sat_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.sub_sat_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u8x16_sub_sat(a: v128, b: v128) -> v128 {
    unsafe { simd_saturating_sub(a.as_u8x16(), b.as_u8x16()).v128() }
}

/// Compares lane-wise signed integers, and returns the minimum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.min_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.min_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_min(a: v128, b: v128) -> v128 {
    let a = a.as_i8x16();
    let b = b.as_i8x16();
    unsafe { simd_select::<simd::i8x16, _>(simd_lt(a, b), a, b).v128() }
}

/// Compares lane-wise unsigned integers, and returns the minimum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.min_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.min_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u8x16_min(a: v128, b: v128) -> v128 {
    let a = a.as_u8x16();
    let b = b.as_u8x16();
    unsafe { simd_select::<simd::i8x16, _>(simd_lt(a, b), a, b).v128() }
}

/// Compares lane-wise signed integers, and returns the maximum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.max_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.max_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i8x16_max(a: v128, b: v128) -> v128 {
    let a = a.as_i8x16();
    let b = b.as_i8x16();
    unsafe { simd_select::<simd::i8x16, _>(simd_gt(a, b), a, b).v128() }
}

/// Compares lane-wise unsigned integers, and returns the maximum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.max_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.max_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u8x16_max(a: v128, b: v128) -> v128 {
    let a = a.as_u8x16();
    let b = b.as_u8x16();
    unsafe { simd_select::<simd::i8x16, _>(simd_gt(a, b), a, b).v128() }
}

/// Lane-wise rounding average.
#[inline]
#[cfg_attr(test, assert_instr(i8x16.avgr_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i8x16.avgr_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u8x16_avgr(a: v128, b: v128) -> v128 {
    unsafe { llvm_avgr_u_i8x16(a.as_i8x16(), b.as_i8x16()).v128() }
}

/// Integer extended pairwise addition producing extended results
/// (twice wider results than the inputs).
#[inline]
#[cfg_attr(test, assert_instr(i16x8.extadd_pairwise_i8x16_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.extadd_pairwise_i8x16_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_extadd_pairwise_i8x16(a: v128) -> v128 {
    unsafe { llvm_i16x8_extadd_pairwise_i8x16_s(a.as_i8x16()).v128() }
}

/// Integer extended pairwise addition producing extended results
/// (twice wider results than the inputs).
#[inline]
#[cfg_attr(test, assert_instr(i16x8.extadd_pairwise_i8x16_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.extadd_pairwise_i8x16_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_extadd_pairwise_u8x16(a: v128) -> v128 {
    unsafe { llvm_i16x8_extadd_pairwise_i8x16_u(a.as_i8x16()).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i16x8_extadd_pairwise_u8x16 as u16x8_extadd_pairwise_u8x16;

/// Lane-wise wrapping absolute value.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.abs))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.abs"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_abs(a: v128) -> v128 {
    let a = a.as_i16x8();
    let zero = simd::i16x8::ZERO;
    unsafe {
        simd_select::<simd::m16x8, simd::i16x8>(simd_lt(a, zero), simd_sub(zero, a), a).v128()
    }
}

/// Negates a 128-bit vectors interpreted as eight 16-bit signed integers
#[inline]
#[cfg_attr(test, assert_instr(i16x8.neg))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.neg"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_neg(a: v128) -> v128 {
    unsafe { simd_mul(a.as_i16x8(), simd::i16x8::splat(-1)).v128() }
}

/// Lane-wise saturating rounding multiplication in Q15 format.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.q15mulr_sat_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.q15mulr_sat_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_q15mulr_sat(a: v128, b: v128) -> v128 {
    unsafe { llvm_q15mulr(a.as_i16x8(), b.as_i16x8()).v128() }
}

/// Returns true if all lanes are non-zero, false otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.all_true))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.all_true"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_all_true(a: v128) -> bool {
    unsafe { llvm_i16x8_all_true(a.as_i16x8()) != 0 }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i16x8_all_true as u16x8_all_true;

/// Extracts the high bit for each lane in `a` and produce a scalar mask with
/// all bits concatenated.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.bitmask))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.bitmask"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_bitmask(a: v128) -> u8 {
    unsafe { llvm_bitmask_i16x8(a.as_i16x8()) as u8 }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i16x8_bitmask as u16x8_bitmask;

/// Converts two input vectors into a smaller lane vector by narrowing each
/// lane.
///
/// Signed saturation to 0x7fff or 0x8000 is used and the input lanes are always
/// interpreted as signed integers.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.narrow_i32x4_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.narrow_i32x4_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_narrow_i32x4(a: v128, b: v128) -> v128 {
    unsafe { llvm_narrow_i16x8_s(a.as_i32x4(), b.as_i32x4()).v128() }
}

/// Converts two input vectors into a smaller lane vector by narrowing each
/// lane.
///
/// Signed saturation to 0x0000 or 0xffff is used and the input lanes are always
/// interpreted as signed integers.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.narrow_i32x4_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.narrow_i32x4_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u16x8_narrow_i32x4(a: v128, b: v128) -> v128 {
    unsafe { llvm_narrow_i16x8_u(a.as_i32x4(), b.as_i32x4()).v128() }
}

/// Converts low half of the smaller lane vector to a larger lane
/// vector, sign extended.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.extend_low_i8x16_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.extend_low_i8x16_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_extend_low_i8x16(a: v128) -> v128 {
    unsafe {
        simd_cast::<simd::i8x8, simd::i16x8>(simd_shuffle!(
            a.as_i8x16(),
            a.as_i8x16(),
            [0, 1, 2, 3, 4, 5, 6, 7],
        ))
        .v128()
    }
}

/// Converts high half of the smaller lane vector to a larger lane
/// vector, sign extended.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.extend_high_i8x16_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.extend_high_i8x16_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_extend_high_i8x16(a: v128) -> v128 {
    unsafe {
        simd_cast::<simd::i8x8, simd::i16x8>(simd_shuffle!(
            a.as_i8x16(),
            a.as_i8x16(),
            [8, 9, 10, 11, 12, 13, 14, 15],
        ))
        .v128()
    }
}

/// Converts low half of the smaller lane vector to a larger lane
/// vector, zero extended.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.extend_low_i8x16_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.extend_low_i8x16_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_extend_low_u8x16(a: v128) -> v128 {
    unsafe {
        simd_cast::<simd::u8x8, simd::u16x8>(simd_shuffle!(
            a.as_u8x16(),
            a.as_u8x16(),
            [0, 1, 2, 3, 4, 5, 6, 7],
        ))
        .v128()
    }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i16x8_extend_low_u8x16 as u16x8_extend_low_u8x16;

/// Converts high half of the smaller lane vector to a larger lane
/// vector, zero extended.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.extend_high_i8x16_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.extend_high_i8x16_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_extend_high_u8x16(a: v128) -> v128 {
    unsafe {
        simd_cast::<simd::u8x8, simd::u16x8>(simd_shuffle!(
            a.as_u8x16(),
            a.as_u8x16(),
            [8, 9, 10, 11, 12, 13, 14, 15],
        ))
        .v128()
    }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i16x8_extend_high_u8x16 as u16x8_extend_high_u8x16;

/// Shifts each lane to the left by the specified number of bits.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.shl))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.shl"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_shl(a: v128, amt: u32) -> v128 {
    // SAFETY: see i8x16_shl for more documentation why this is unsafe,
    // essentially the shift amount must be valid hence the mask.
    unsafe { simd_shl(a.as_i16x8(), simd::i16x8::splat((amt & 0xf) as i16)).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i16x8_shl as u16x8_shl;

/// Shifts each lane to the right by the specified number of bits, sign
/// extending.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.shr_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.shr_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_shr(a: v128, amt: u32) -> v128 {
    // SAFETY: see i8x16_shl for more documentation why this is unsafe,
    // essentially the shift amount must be valid hence the mask.
    unsafe { simd_shr(a.as_i16x8(), simd::i16x8::splat((amt & 0xf) as i16)).v128() }
}

/// Shifts each lane to the right by the specified number of bits, shifting in
/// zeros.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.shr_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.shr_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u16x8_shr(a: v128, amt: u32) -> v128 {
    // SAFETY: see i8x16_shl for more documentation why this is unsafe,
    // essentially the shift amount must be valid hence the mask.
    unsafe { simd_shr(a.as_u16x8(), simd::u16x8::splat((amt & 0xf) as u16)).v128() }
}

/// Adds two 128-bit vectors as if they were two packed eight 16-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.add))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.add"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_add(a: v128, b: v128) -> v128 {
    unsafe { simd_add(a.as_i16x8(), b.as_i16x8()).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i16x8_add as u16x8_add;

/// Adds two 128-bit vectors as if they were two packed eight 16-bit signed
/// integers, saturating on overflow to `i16::MAX`.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.add_sat_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.add_sat_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_add_sat(a: v128, b: v128) -> v128 {
    unsafe { simd_saturating_add(a.as_i16x8(), b.as_i16x8()).v128() }
}

/// Adds two 128-bit vectors as if they were two packed eight 16-bit unsigned
/// integers, saturating on overflow to `u16::MAX`.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.add_sat_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.add_sat_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u16x8_add_sat(a: v128, b: v128) -> v128 {
    unsafe { simd_saturating_add(a.as_u16x8(), b.as_u16x8()).v128() }
}

/// Subtracts two 128-bit vectors as if they were two packed eight 16-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.sub))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.sub"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_sub(a: v128, b: v128) -> v128 {
    unsafe { simd_sub(a.as_i16x8(), b.as_i16x8()).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i16x8_sub as u16x8_sub;

/// Subtracts two 128-bit vectors as if they were two packed eight 16-bit
/// signed integers, saturating on overflow to `i16::MIN`.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.sub_sat_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.sub_sat_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_sub_sat(a: v128, b: v128) -> v128 {
    unsafe { simd_saturating_sub(a.as_i16x8(), b.as_i16x8()).v128() }
}

/// Subtracts two 128-bit vectors as if they were two packed eight 16-bit
/// unsigned integers, saturating on overflow to 0.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.sub_sat_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.sub_sat_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u16x8_sub_sat(a: v128, b: v128) -> v128 {
    unsafe { simd_saturating_sub(a.as_u16x8(), b.as_u16x8()).v128() }
}

/// Multiplies two 128-bit vectors as if they were two packed eight 16-bit
/// signed integers.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.mul))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.mul"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_mul(a: v128, b: v128) -> v128 {
    unsafe { simd_mul(a.as_i16x8(), b.as_i16x8()).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i16x8_mul as u16x8_mul;

/// Compares lane-wise signed integers, and returns the minimum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.min_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.min_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_min(a: v128, b: v128) -> v128 {
    let a = a.as_i16x8();
    let b = b.as_i16x8();
    unsafe { simd_select::<simd::i16x8, _>(simd_lt(a, b), a, b).v128() }
}

/// Compares lane-wise unsigned integers, and returns the minimum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.min_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.min_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u16x8_min(a: v128, b: v128) -> v128 {
    let a = a.as_u16x8();
    let b = b.as_u16x8();
    unsafe { simd_select::<simd::i16x8, _>(simd_lt(a, b), a, b).v128() }
}

/// Compares lane-wise signed integers, and returns the maximum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.max_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.max_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_max(a: v128, b: v128) -> v128 {
    let a = a.as_i16x8();
    let b = b.as_i16x8();
    unsafe { simd_select::<simd::i16x8, _>(simd_gt(a, b), a, b).v128() }
}

/// Compares lane-wise unsigned integers, and returns the maximum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.max_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.max_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u16x8_max(a: v128, b: v128) -> v128 {
    let a = a.as_u16x8();
    let b = b.as_u16x8();
    unsafe { simd_select::<simd::i16x8, _>(simd_gt(a, b), a, b).v128() }
}

/// Lane-wise rounding average.
#[inline]
#[cfg_attr(test, assert_instr(i16x8.avgr_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.avgr_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u16x8_avgr(a: v128, b: v128) -> v128 {
    unsafe { llvm_avgr_u_i16x8(a.as_i16x8(), b.as_i16x8()).v128() }
}

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i16x8_mul(i16x8_extend_low_i8x16(a), i16x8_extend_low_i8x16(b))`
#[inline]
#[cfg_attr(test, assert_instr(i16x8.extmul_low_i8x16_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.extmul_low_i8x16_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_extmul_low_i8x16(a: v128, b: v128) -> v128 {
    unsafe {
        let lhs = simd_cast::<simd::i8x8, simd::i16x8>(simd_shuffle!(
            a.as_i8x16(),
            a.as_i8x16(),
            [0, 1, 2, 3, 4, 5, 6, 7],
        ));
        let rhs = simd_cast::<simd::i8x8, simd::i16x8>(simd_shuffle!(
            b.as_i8x16(),
            b.as_i8x16(),
            [0, 1, 2, 3, 4, 5, 6, 7],
        ));
        simd_mul(lhs, rhs).v128()
    }
}

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i16x8_mul(i16x8_extend_high_i8x16(a), i16x8_extend_high_i8x16(b))`
#[inline]
#[cfg_attr(test, assert_instr(i16x8.extmul_high_i8x16_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.extmul_high_i8x16_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_extmul_high_i8x16(a: v128, b: v128) -> v128 {
    unsafe {
        let lhs = simd_cast::<simd::i8x8, simd::i16x8>(simd_shuffle!(
            a.as_i8x16(),
            a.as_i8x16(),
            [8, 9, 10, 11, 12, 13, 14, 15],
        ));
        let rhs = simd_cast::<simd::i8x8, simd::i16x8>(simd_shuffle!(
            b.as_i8x16(),
            b.as_i8x16(),
            [8, 9, 10, 11, 12, 13, 14, 15],
        ));
        simd_mul(lhs, rhs).v128()
    }
}

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i16x8_mul(i16x8_extend_low_u8x16(a), i16x8_extend_low_u8x16(b))`
#[inline]
#[cfg_attr(test, assert_instr(i16x8.extmul_low_i8x16_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.extmul_low_i8x16_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_extmul_low_u8x16(a: v128, b: v128) -> v128 {
    unsafe {
        let lhs = simd_cast::<simd::u8x8, simd::u16x8>(simd_shuffle!(
            a.as_u8x16(),
            a.as_u8x16(),
            [0, 1, 2, 3, 4, 5, 6, 7],
        ));
        let rhs = simd_cast::<simd::u8x8, simd::u16x8>(simd_shuffle!(
            b.as_u8x16(),
            b.as_u8x16(),
            [0, 1, 2, 3, 4, 5, 6, 7],
        ));
        simd_mul(lhs, rhs).v128()
    }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i16x8_extmul_low_u8x16 as u16x8_extmul_low_u8x16;

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i16x8_mul(i16x8_extend_high_u8x16(a), i16x8_extend_high_u8x16(b))`
#[inline]
#[cfg_attr(test, assert_instr(i16x8.extmul_high_i8x16_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i16x8.extmul_high_i8x16_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i16x8_extmul_high_u8x16(a: v128, b: v128) -> v128 {
    unsafe {
        let lhs = simd_cast::<simd::u8x8, simd::u16x8>(simd_shuffle!(
            a.as_u8x16(),
            a.as_u8x16(),
            [8, 9, 10, 11, 12, 13, 14, 15],
        ));
        let rhs = simd_cast::<simd::u8x8, simd::u16x8>(simd_shuffle!(
            b.as_u8x16(),
            b.as_u8x16(),
            [8, 9, 10, 11, 12, 13, 14, 15],
        ));
        simd_mul(lhs, rhs).v128()
    }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i16x8_extmul_high_u8x16 as u16x8_extmul_high_u8x16;

/// Integer extended pairwise addition producing extended results
/// (twice wider results than the inputs).
#[inline]
#[cfg_attr(test, assert_instr(i32x4.extadd_pairwise_i16x8_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.extadd_pairwise_i16x8_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_extadd_pairwise_i16x8(a: v128) -> v128 {
    unsafe { llvm_i32x4_extadd_pairwise_i16x8_s(a.as_i16x8()).v128() }
}

/// Integer extended pairwise addition producing extended results
/// (twice wider results than the inputs).
#[inline]
#[cfg_attr(test, assert_instr(i32x4.extadd_pairwise_i16x8_u))]
#[doc(alias("i32x4.extadd_pairwise_i16x8_u"))]
#[target_feature(enable = "simd128")]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_extadd_pairwise_u16x8(a: v128) -> v128 {
    unsafe { llvm_i32x4_extadd_pairwise_i16x8_u(a.as_i16x8()).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i32x4_extadd_pairwise_u16x8 as u32x4_extadd_pairwise_u16x8;

/// Lane-wise wrapping absolute value.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.abs))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.abs"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_abs(a: v128) -> v128 {
    let a = a.as_i32x4();
    let zero = simd::i32x4::ZERO;
    unsafe {
        simd_select::<simd::m32x4, simd::i32x4>(simd_lt(a, zero), simd_sub(zero, a), a).v128()
    }
}

/// Negates a 128-bit vectors interpreted as four 32-bit signed integers
#[inline]
#[cfg_attr(test, assert_instr(i32x4.neg))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.neg"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_neg(a: v128) -> v128 {
    unsafe { simd_mul(a.as_i32x4(), simd::i32x4::splat(-1)).v128() }
}

/// Returns true if all lanes are non-zero, false otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.all_true))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.all_true"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_all_true(a: v128) -> bool {
    unsafe { llvm_i32x4_all_true(a.as_i32x4()) != 0 }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i32x4_all_true as u32x4_all_true;

/// Extracts the high bit for each lane in `a` and produce a scalar mask with
/// all bits concatenated.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.bitmask))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.bitmask"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_bitmask(a: v128) -> u8 {
    unsafe { llvm_bitmask_i32x4(a.as_i32x4()) as u8 }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i32x4_bitmask as u32x4_bitmask;

/// Converts low half of the smaller lane vector to a larger lane
/// vector, sign extended.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.extend_low_i16x8_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.extend_low_i16x8_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_extend_low_i16x8(a: v128) -> v128 {
    unsafe {
        simd_cast::<simd::i16x4, simd::i32x4>(simd_shuffle!(
            a.as_i16x8(),
            a.as_i16x8(),
            [0, 1, 2, 3]
        ))
        .v128()
    }
}

/// Converts high half of the smaller lane vector to a larger lane
/// vector, sign extended.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.extend_high_i16x8_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.extend_high_i16x8_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_extend_high_i16x8(a: v128) -> v128 {
    unsafe {
        simd_cast::<simd::i16x4, simd::i32x4>(simd_shuffle!(
            a.as_i16x8(),
            a.as_i16x8(),
            [4, 5, 6, 7]
        ))
        .v128()
    }
}

/// Converts low half of the smaller lane vector to a larger lane
/// vector, zero extended.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.extend_low_i16x8_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.extend_low_i16x8_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_extend_low_u16x8(a: v128) -> v128 {
    unsafe {
        simd_cast::<simd::u16x4, simd::u32x4>(simd_shuffle!(
            a.as_u16x8(),
            a.as_u16x8(),
            [0, 1, 2, 3]
        ))
        .v128()
    }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i32x4_extend_low_u16x8 as u32x4_extend_low_u16x8;

/// Converts high half of the smaller lane vector to a larger lane
/// vector, zero extended.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.extend_high_i16x8_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.extend_high_i16x8_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_extend_high_u16x8(a: v128) -> v128 {
    unsafe {
        simd_cast::<simd::u16x4, simd::u32x4>(simd_shuffle!(
            a.as_u16x8(),
            a.as_u16x8(),
            [4, 5, 6, 7]
        ))
        .v128()
    }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i32x4_extend_high_u16x8 as u32x4_extend_high_u16x8;

/// Shifts each lane to the left by the specified number of bits.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.shl))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.shl"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_shl(a: v128, amt: u32) -> v128 {
    // SAFETY: see i8x16_shl for more documentation why this is unsafe,
    // essentially the shift amount must be valid hence the mask.
    unsafe { simd_shl(a.as_i32x4(), simd::i32x4::splat((amt & 0x1f) as i32)).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i32x4_shl as u32x4_shl;

/// Shifts each lane to the right by the specified number of bits, sign
/// extending.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.shr_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.shr_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_shr(a: v128, amt: u32) -> v128 {
    // SAFETY: see i8x16_shl for more documentation why this is unsafe,
    // essentially the shift amount must be valid hence the mask.
    unsafe { simd_shr(a.as_i32x4(), simd::i32x4::splat((amt & 0x1f) as i32)).v128() }
}

/// Shifts each lane to the right by the specified number of bits, shifting in
/// zeros.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.shr_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.shr_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u32x4_shr(a: v128, amt: u32) -> v128 {
    // SAFETY: see i8x16_shl for more documentation why this is unsafe,
    // essentially the shift amount must be valid hence the mask.
    unsafe { simd_shr(a.as_u32x4(), simd::u32x4::splat(amt & 0x1f)).v128() }
}

/// Adds two 128-bit vectors as if they were two packed four 32-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.add))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.add"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_add(a: v128, b: v128) -> v128 {
    unsafe { simd_add(a.as_i32x4(), b.as_i32x4()).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i32x4_add as u32x4_add;

/// Subtracts two 128-bit vectors as if they were two packed four 32-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.sub))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.sub"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_sub(a: v128, b: v128) -> v128 {
    unsafe { simd_sub(a.as_i32x4(), b.as_i32x4()).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i32x4_sub as u32x4_sub;

/// Multiplies two 128-bit vectors as if they were two packed four 32-bit
/// signed integers.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.mul))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.mul"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_mul(a: v128, b: v128) -> v128 {
    unsafe { simd_mul(a.as_i32x4(), b.as_i32x4()).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i32x4_mul as u32x4_mul;

/// Compares lane-wise signed integers, and returns the minimum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.min_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.min_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_min(a: v128, b: v128) -> v128 {
    let a = a.as_i32x4();
    let b = b.as_i32x4();
    unsafe { simd_select::<simd::i32x4, _>(simd_lt(a, b), a, b).v128() }
}

/// Compares lane-wise unsigned integers, and returns the minimum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.min_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.min_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u32x4_min(a: v128, b: v128) -> v128 {
    let a = a.as_u32x4();
    let b = b.as_u32x4();
    unsafe { simd_select::<simd::i32x4, _>(simd_lt(a, b), a, b).v128() }
}

/// Compares lane-wise signed integers, and returns the maximum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.max_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.max_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_max(a: v128, b: v128) -> v128 {
    let a = a.as_i32x4();
    let b = b.as_i32x4();
    unsafe { simd_select::<simd::i32x4, _>(simd_gt(a, b), a, b).v128() }
}

/// Compares lane-wise unsigned integers, and returns the maximum of
/// each pair.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.max_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.max_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u32x4_max(a: v128, b: v128) -> v128 {
    let a = a.as_u32x4();
    let b = b.as_u32x4();
    unsafe { simd_select::<simd::i32x4, _>(simd_gt(a, b), a, b).v128() }
}

/// Lane-wise multiply signed 16-bit integers in the two input vectors and add
/// adjacent pairs of the full 32-bit results.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.dot_i16x8_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.dot_i16x8_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_dot_i16x8(a: v128, b: v128) -> v128 {
    unsafe { llvm_i32x4_dot_i16x8_s(a.as_i16x8(), b.as_i16x8()).v128() }
}

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i32x4_mul(i32x4_extend_low_i16x8_s(a), i32x4_extend_low_i16x8_s(b))`
#[inline]
#[cfg_attr(test, assert_instr(i32x4.extmul_low_i16x8_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.extmul_low_i16x8_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_extmul_low_i16x8(a: v128, b: v128) -> v128 {
    unsafe {
        let lhs = simd_cast::<simd::i16x4, simd::i32x4>(simd_shuffle!(
            a.as_i16x8(),
            a.as_i16x8(),
            [0, 1, 2, 3]
        ));
        let rhs = simd_cast::<simd::i16x4, simd::i32x4>(simd_shuffle!(
            b.as_i16x8(),
            b.as_i16x8(),
            [0, 1, 2, 3]
        ));
        simd_mul(lhs, rhs).v128()
    }
}

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i32x4_mul(i32x4_extend_high_i16x8_s(a), i32x4_extend_high_i16x8_s(b))`
#[inline]
#[cfg_attr(test, assert_instr(i32x4.extmul_high_i16x8_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.extmul_high_i16x8_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_extmul_high_i16x8(a: v128, b: v128) -> v128 {
    unsafe {
        let lhs = simd_cast::<simd::i16x4, simd::i32x4>(simd_shuffle!(
            a.as_i16x8(),
            a.as_i16x8(),
            [4, 5, 6, 7]
        ));
        let rhs = simd_cast::<simd::i16x4, simd::i32x4>(simd_shuffle!(
            b.as_i16x8(),
            b.as_i16x8(),
            [4, 5, 6, 7]
        ));
        simd_mul(lhs, rhs).v128()
    }
}

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i32x4_mul(i32x4_extend_low_u16x8(a), i32x4_extend_low_u16x8(b))`
#[inline]
#[cfg_attr(test, assert_instr(i32x4.extmul_low_i16x8_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.extmul_low_i16x8_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_extmul_low_u16x8(a: v128, b: v128) -> v128 {
    unsafe {
        let lhs = simd_cast::<simd::u16x4, simd::u32x4>(simd_shuffle!(
            a.as_u16x8(),
            a.as_u16x8(),
            [0, 1, 2, 3]
        ));
        let rhs = simd_cast::<simd::u16x4, simd::u32x4>(simd_shuffle!(
            b.as_u16x8(),
            b.as_u16x8(),
            [0, 1, 2, 3]
        ));
        simd_mul(lhs, rhs).v128()
    }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i32x4_extmul_low_u16x8 as u32x4_extmul_low_u16x8;

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i32x4_mul(i32x4_extend_high_u16x8(a), i32x4_extend_high_u16x8(b))`
#[inline]
#[cfg_attr(test, assert_instr(i32x4.extmul_high_i16x8_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.extmul_high_i16x8_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_extmul_high_u16x8(a: v128, b: v128) -> v128 {
    unsafe {
        let lhs = simd_cast::<simd::u16x4, simd::u32x4>(simd_shuffle!(
            a.as_u16x8(),
            a.as_u16x8(),
            [4, 5, 6, 7]
        ));
        let rhs = simd_cast::<simd::u16x4, simd::u32x4>(simd_shuffle!(
            b.as_u16x8(),
            b.as_u16x8(),
            [4, 5, 6, 7]
        ));
        simd_mul(lhs, rhs).v128()
    }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i32x4_extmul_high_u16x8 as u32x4_extmul_high_u16x8;

/// Lane-wise wrapping absolute value.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.abs))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.abs"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_abs(a: v128) -> v128 {
    let a = a.as_i64x2();
    let zero = simd::i64x2::ZERO;
    unsafe {
        simd_select::<simd::m64x2, simd::i64x2>(simd_lt(a, zero), simd_sub(zero, a), a).v128()
    }
}

/// Negates a 128-bit vectors interpreted as two 64-bit signed integers
#[inline]
#[cfg_attr(test, assert_instr(i64x2.neg))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.neg"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_neg(a: v128) -> v128 {
    unsafe { simd_mul(a.as_i64x2(), simd::i64x2::splat(-1)).v128() }
}

/// Returns true if all lanes are non-zero, false otherwise.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.all_true))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.all_true"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_all_true(a: v128) -> bool {
    unsafe { llvm_i64x2_all_true(a.as_i64x2()) != 0 }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i64x2_all_true as u64x2_all_true;

/// Extracts the high bit for each lane in `a` and produce a scalar mask with
/// all bits concatenated.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.bitmask))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.bitmask"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_bitmask(a: v128) -> u8 {
    unsafe { llvm_bitmask_i64x2(a.as_i64x2()) as u8 }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i64x2_bitmask as u64x2_bitmask;

/// Converts low half of the smaller lane vector to a larger lane
/// vector, sign extended.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.extend_low_i32x4_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.extend_low_i32x4_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_extend_low_i32x4(a: v128) -> v128 {
    unsafe {
        simd_cast::<simd::i32x2, simd::i64x2>(simd_shuffle!(a.as_i32x4(), a.as_i32x4(), [0, 1]))
            .v128()
    }
}

/// Converts high half of the smaller lane vector to a larger lane
/// vector, sign extended.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.extend_high_i32x4_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.extend_high_i32x4_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_extend_high_i32x4(a: v128) -> v128 {
    unsafe {
        simd_cast::<simd::i32x2, simd::i64x2>(simd_shuffle!(a.as_i32x4(), a.as_i32x4(), [2, 3]))
            .v128()
    }
}

/// Converts low half of the smaller lane vector to a larger lane
/// vector, zero extended.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.extend_low_i32x4_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.extend_low_i32x4_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_extend_low_u32x4(a: v128) -> v128 {
    unsafe {
        simd_cast::<simd::u32x2, simd::i64x2>(simd_shuffle!(a.as_u32x4(), a.as_u32x4(), [0, 1]))
            .v128()
    }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i64x2_extend_low_u32x4 as u64x2_extend_low_u32x4;

/// Converts high half of the smaller lane vector to a larger lane
/// vector, zero extended.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.extend_high_i32x4_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.extend_high_i32x4_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_extend_high_u32x4(a: v128) -> v128 {
    unsafe {
        simd_cast::<simd::u32x2, simd::i64x2>(simd_shuffle!(a.as_u32x4(), a.as_u32x4(), [2, 3]))
            .v128()
    }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i64x2_extend_high_u32x4 as u64x2_extend_high_u32x4;

/// Shifts each lane to the left by the specified number of bits.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.shl))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.shl"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_shl(a: v128, amt: u32) -> v128 {
    // SAFETY: see i8x16_shl for more documentation why this is unsafe,
    // essentially the shift amount must be valid hence the mask.
    unsafe { simd_shl(a.as_i64x2(), simd::i64x2::splat((amt & 0x3f) as i64)).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i64x2_shl as u64x2_shl;

/// Shifts each lane to the right by the specified number of bits, sign
/// extending.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.shr_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.shr_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_shr(a: v128, amt: u32) -> v128 {
    // SAFETY: see i8x16_shl for more documentation why this is unsafe,
    // essentially the shift amount must be valid hence the mask.
    unsafe { simd_shr(a.as_i64x2(), simd::i64x2::splat((amt & 0x3f) as i64)).v128() }
}

/// Shifts each lane to the right by the specified number of bits, shifting in
/// zeros.
///
/// Only the low bits of the shift amount are used if the shift amount is
/// greater than the lane width.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.shr_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.shr_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u64x2_shr(a: v128, amt: u32) -> v128 {
    // SAFETY: see i8x16_shl for more documentation why this is unsafe,
    // essentially the shift amount must be valid hence the mask.
    unsafe { simd_shr(a.as_u64x2(), simd::u64x2::splat((amt & 0x3f) as u64)).v128() }
}

/// Adds two 128-bit vectors as if they were two packed two 64-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.add))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.add"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_add(a: v128, b: v128) -> v128 {
    unsafe { simd_add(a.as_i64x2(), b.as_i64x2()).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i64x2_add as u64x2_add;

/// Subtracts two 128-bit vectors as if they were two packed two 64-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.sub))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.sub"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_sub(a: v128, b: v128) -> v128 {
    unsafe { simd_sub(a.as_i64x2(), b.as_i64x2()).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i64x2_sub as u64x2_sub;

/// Multiplies two 128-bit vectors as if they were two packed two 64-bit integers.
#[inline]
#[cfg_attr(test, assert_instr(i64x2.mul))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.mul"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_mul(a: v128, b: v128) -> v128 {
    unsafe { simd_mul(a.as_i64x2(), b.as_i64x2()).v128() }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i64x2_mul as u64x2_mul;

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i64x2_mul(i64x2_extend_low_i32x4_s(a), i64x2_extend_low_i32x4_s(b))`
#[inline]
#[cfg_attr(test, assert_instr(i64x2.extmul_low_i32x4_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.extmul_low_i32x4_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_extmul_low_i32x4(a: v128, b: v128) -> v128 {
    unsafe {
        let lhs = simd_cast::<simd::i32x2, simd::i64x2>(simd_shuffle!(
            a.as_i32x4(),
            a.as_i32x4(),
            [0, 1]
        ));
        let rhs = simd_cast::<simd::i32x2, simd::i64x2>(simd_shuffle!(
            b.as_i32x4(),
            b.as_i32x4(),
            [0, 1]
        ));
        simd_mul(lhs, rhs).v128()
    }
}

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i64x2_mul(i64x2_extend_high_i32x4_s(a), i64x2_extend_high_i32x4_s(b))`
#[inline]
#[cfg_attr(test, assert_instr(i64x2.extmul_high_i32x4_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.extmul_high_i32x4_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_extmul_high_i32x4(a: v128, b: v128) -> v128 {
    unsafe {
        let lhs = simd_cast::<simd::i32x2, simd::i64x2>(simd_shuffle!(
            a.as_i32x4(),
            a.as_i32x4(),
            [2, 3]
        ));
        let rhs = simd_cast::<simd::i32x2, simd::i64x2>(simd_shuffle!(
            b.as_i32x4(),
            b.as_i32x4(),
            [2, 3]
        ));
        simd_mul(lhs, rhs).v128()
    }
}

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i64x2_mul(i64x2_extend_low_i32x4_u(a), i64x2_extend_low_i32x4_u(b))`
#[inline]
#[cfg_attr(test, assert_instr(i64x2.extmul_low_i32x4_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.extmul_low_i32x4_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_extmul_low_u32x4(a: v128, b: v128) -> v128 {
    unsafe {
        let lhs = simd_cast::<simd::u32x2, simd::u64x2>(simd_shuffle!(
            a.as_u32x4(),
            a.as_u32x4(),
            [0, 1]
        ));
        let rhs = simd_cast::<simd::u32x2, simd::u64x2>(simd_shuffle!(
            b.as_u32x4(),
            b.as_u32x4(),
            [0, 1]
        ));
        simd_mul(lhs, rhs).v128()
    }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i64x2_extmul_low_u32x4 as u64x2_extmul_low_u32x4;

/// Lane-wise integer extended multiplication producing twice wider result than
/// the inputs.
///
/// Equivalent of `i64x2_mul(i64x2_extend_high_i32x4_u(a), i64x2_extend_high_i32x4_u(b))`
#[inline]
#[cfg_attr(test, assert_instr(i64x2.extmul_high_i32x4_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i64x2.extmul_high_i32x4_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i64x2_extmul_high_u32x4(a: v128, b: v128) -> v128 {
    unsafe {
        let lhs = simd_cast::<simd::u32x2, simd::u64x2>(simd_shuffle!(
            a.as_u32x4(),
            a.as_u32x4(),
            [2, 3]
        ));
        let rhs = simd_cast::<simd::u32x2, simd::u64x2>(simd_shuffle!(
            b.as_u32x4(),
            b.as_u32x4(),
            [2, 3]
        ));
        simd_mul(lhs, rhs).v128()
    }
}

#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use i64x2_extmul_high_u32x4 as u64x2_extmul_high_u32x4;

/// Lane-wise rounding to the nearest integral value not smaller than the input.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.ceil))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.ceil"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_ceil(a: v128) -> v128 {
    unsafe { simd_ceil(a.as_f32x4()).v128() }
}

/// Lane-wise rounding to the nearest integral value not greater than the input.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.floor))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.floor"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_floor(a: v128) -> v128 {
    unsafe { simd_floor(a.as_f32x4()).v128() }
}

/// Lane-wise rounding to the nearest integral value with the magnitude not
/// larger than the input.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.trunc))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.trunc"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_trunc(a: v128) -> v128 {
    unsafe { simd_trunc(a.as_f32x4()).v128() }
}

/// Lane-wise rounding to the nearest integral value; if two values are equally
/// near, rounds to the even one.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.nearest))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.nearest"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_nearest(a: v128) -> v128 {
    unsafe { llvm_f32x4_nearest(a.as_f32x4()).v128() }
}

/// Calculates the absolute value of each lane of a 128-bit vector interpreted
/// as four 32-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.abs))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.abs"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_abs(a: v128) -> v128 {
    unsafe { simd_fabs(a.as_f32x4()).v128() }
}

/// Negates each lane of a 128-bit vector interpreted as four 32-bit floating
/// point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.neg))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.neg"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_neg(a: v128) -> v128 {
    unsafe { simd_neg(a.as_f32x4()).v128() }
}

/// Calculates the square root of each lane of a 128-bit vector interpreted as
/// four 32-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.sqrt))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.sqrt"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_sqrt(a: v128) -> v128 {
    unsafe { simd_fsqrt(a.as_f32x4()).v128() }
}

/// Lane-wise addition of two 128-bit vectors interpreted as four 32-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.add))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.add"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_add(a: v128, b: v128) -> v128 {
    unsafe { simd_add(a.as_f32x4(), b.as_f32x4()).v128() }
}

/// Lane-wise subtraction of two 128-bit vectors interpreted as four 32-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.sub))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.sub"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_sub(a: v128, b: v128) -> v128 {
    unsafe { simd_sub(a.as_f32x4(), b.as_f32x4()).v128() }
}

/// Lane-wise multiplication of two 128-bit vectors interpreted as four 32-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.mul))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.mul"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_mul(a: v128, b: v128) -> v128 {
    unsafe { simd_mul(a.as_f32x4(), b.as_f32x4()).v128() }
}

/// Lane-wise division of two 128-bit vectors interpreted as four 32-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.div))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.div"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_div(a: v128, b: v128) -> v128 {
    unsafe { simd_div(a.as_f32x4(), b.as_f32x4()).v128() }
}

/// Calculates the lane-wise minimum of two 128-bit vectors interpreted
/// as four 32-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.min))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.min"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_min(a: v128, b: v128) -> v128 {
    unsafe { llvm_f32x4_min(a.as_f32x4(), b.as_f32x4()).v128() }
}

/// Calculates the lane-wise minimum of two 128-bit vectors interpreted
/// as four 32-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.max))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.max"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_max(a: v128, b: v128) -> v128 {
    unsafe { llvm_f32x4_max(a.as_f32x4(), b.as_f32x4()).v128() }
}

/// Lane-wise minimum value, defined as `b < a ? b : a`
#[inline]
#[cfg_attr(test, assert_instr(f32x4.pmin))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.pmin"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_pmin(a: v128, b: v128) -> v128 {
    unsafe {
        simd_select::<simd::m32x4, simd::f32x4>(
            simd_lt(b.as_f32x4(), a.as_f32x4()),
            b.as_f32x4(),
            a.as_f32x4(),
        )
        .v128()
    }
}

/// Lane-wise maximum value, defined as `a < b ? b : a`
#[inline]
#[cfg_attr(test, assert_instr(f32x4.pmax))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.pmax"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_pmax(a: v128, b: v128) -> v128 {
    unsafe {
        simd_select::<simd::m32x4, simd::f32x4>(
            simd_lt(a.as_f32x4(), b.as_f32x4()),
            b.as_f32x4(),
            a.as_f32x4(),
        )
        .v128()
    }
}

/// Lane-wise rounding to the nearest integral value not smaller than the input.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.ceil))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.ceil"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_ceil(a: v128) -> v128 {
    unsafe { simd_ceil(a.as_f64x2()).v128() }
}

/// Lane-wise rounding to the nearest integral value not greater than the input.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.floor))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.floor"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_floor(a: v128) -> v128 {
    unsafe { simd_floor(a.as_f64x2()).v128() }
}

/// Lane-wise rounding to the nearest integral value with the magnitude not
/// larger than the input.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.trunc))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.trunc"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_trunc(a: v128) -> v128 {
    unsafe { simd_trunc(a.as_f64x2()).v128() }
}

/// Lane-wise rounding to the nearest integral value; if two values are equally
/// near, rounds to the even one.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.nearest))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.nearest"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_nearest(a: v128) -> v128 {
    unsafe { llvm_f64x2_nearest(a.as_f64x2()).v128() }
}

/// Calculates the absolute value of each lane of a 128-bit vector interpreted
/// as two 64-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.abs))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.abs"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_abs(a: v128) -> v128 {
    unsafe { simd_fabs(a.as_f64x2()).v128() }
}

/// Negates each lane of a 128-bit vector interpreted as two 64-bit floating
/// point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.neg))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.neg"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_neg(a: v128) -> v128 {
    unsafe { simd_neg(a.as_f64x2()).v128() }
}

/// Calculates the square root of each lane of a 128-bit vector interpreted as
/// two 64-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.sqrt))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.sqrt"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_sqrt(a: v128) -> v128 {
    unsafe { simd_fsqrt(a.as_f64x2()).v128() }
}

/// Lane-wise add of two 128-bit vectors interpreted as two 64-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.add))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.add"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_add(a: v128, b: v128) -> v128 {
    unsafe { simd_add(a.as_f64x2(), b.as_f64x2()).v128() }
}

/// Lane-wise subtract of two 128-bit vectors interpreted as two 64-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.sub))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.sub"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_sub(a: v128, b: v128) -> v128 {
    unsafe { simd_sub(a.as_f64x2(), b.as_f64x2()).v128() }
}

/// Lane-wise multiply of two 128-bit vectors interpreted as two 64-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.mul))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.mul"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_mul(a: v128, b: v128) -> v128 {
    unsafe { simd_mul(a.as_f64x2(), b.as_f64x2()).v128() }
}

/// Lane-wise divide of two 128-bit vectors interpreted as two 64-bit
/// floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.div))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.div"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_div(a: v128, b: v128) -> v128 {
    unsafe { simd_div(a.as_f64x2(), b.as_f64x2()).v128() }
}

/// Calculates the lane-wise minimum of two 128-bit vectors interpreted
/// as two 64-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.min))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.min"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_min(a: v128, b: v128) -> v128 {
    unsafe { llvm_f64x2_min(a.as_f64x2(), b.as_f64x2()).v128() }
}

/// Calculates the lane-wise maximum of two 128-bit vectors interpreted
/// as two 64-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.max))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.max"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_max(a: v128, b: v128) -> v128 {
    unsafe { llvm_f64x2_max(a.as_f64x2(), b.as_f64x2()).v128() }
}

/// Lane-wise minimum value, defined as `b < a ? b : a`
#[inline]
#[cfg_attr(test, assert_instr(f64x2.pmin))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.pmin"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_pmin(a: v128, b: v128) -> v128 {
    unsafe {
        simd_select::<simd::m64x2, simd::f64x2>(
            simd_lt(b.as_f64x2(), a.as_f64x2()),
            b.as_f64x2(),
            a.as_f64x2(),
        )
        .v128()
    }
}

/// Lane-wise maximum value, defined as `a < b ? b : a`
#[inline]
#[cfg_attr(test, assert_instr(f64x2.pmax))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.pmax"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_pmax(a: v128, b: v128) -> v128 {
    unsafe {
        simd_select::<simd::m64x2, simd::f64x2>(
            simd_lt(a.as_f64x2(), b.as_f64x2()),
            b.as_f64x2(),
            a.as_f64x2(),
        )
        .v128()
    }
}

/// Converts a 128-bit vector interpreted as four 32-bit floating point numbers
/// into a 128-bit vector of four 32-bit signed integers.
///
/// NaN is converted to 0 and if it's out of bounds it becomes the nearest
/// representable intger.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.trunc_sat_f32x4_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.trunc_sat_f32x4_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_trunc_sat_f32x4(a: v128) -> v128 {
    unsafe { simd_as::<simd::f32x4, simd::i32x4>(a.as_f32x4()).v128() }
}

/// Converts a 128-bit vector interpreted as four 32-bit floating point numbers
/// into a 128-bit vector of four 32-bit unsigned integers.
///
/// NaN is converted to 0 and if it's out of bounds it becomes the nearest
/// representable intger.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.trunc_sat_f32x4_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.trunc_sat_f32x4_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u32x4_trunc_sat_f32x4(a: v128) -> v128 {
    unsafe { simd_as::<simd::f32x4, simd::u32x4>(a.as_f32x4()).v128() }
}

/// Converts a 128-bit vector interpreted as four 32-bit signed integers into a
/// 128-bit vector of four 32-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.convert_i32x4_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.convert_i32x4_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_convert_i32x4(a: v128) -> v128 {
    unsafe { simd_cast::<_, simd::f32x4>(a.as_i32x4()).v128() }
}

/// Converts a 128-bit vector interpreted as four 32-bit unsigned integers into a
/// 128-bit vector of four 32-bit floating point numbers.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.convert_i32x4_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.convert_i32x4_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_convert_u32x4(a: v128) -> v128 {
    unsafe { simd_cast::<_, simd::f32x4>(a.as_u32x4()).v128() }
}

/// Saturating conversion of the two double-precision floating point lanes to
/// two lower integer lanes using the IEEE `convertToIntegerTowardZero`
/// function.
///
/// The two higher lanes of the result are initialized to zero. If any input
/// lane is a NaN, the resulting lane is 0. If the rounded integer value of a
/// lane is outside the range of the destination type, the result is saturated
/// to the nearest representable integer value.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.trunc_sat_f64x2_s_zero))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.trunc_sat_f64x2_s_zero"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn i32x4_trunc_sat_f64x2_zero(a: v128) -> v128 {
    let ret: simd::i32x4 = unsafe {
        simd_shuffle!(
            simd_as::<simd::f64x2, simd::i32x2>(a.as_f64x2()),
            simd::i32x2::ZERO,
            [0, 1, 2, 3],
        )
    };
    ret.v128()
}

/// Saturating conversion of the two double-precision floating point lanes to
/// two lower integer lanes using the IEEE `convertToIntegerTowardZero`
/// function.
///
/// The two higher lanes of the result are initialized to zero. If any input
/// lane is a NaN, the resulting lane is 0. If the rounded integer value of a
/// lane is outside the range of the destination type, the result is saturated
/// to the nearest representable integer value.
#[inline]
#[cfg_attr(test, assert_instr(i32x4.trunc_sat_f64x2_u_zero))]
#[target_feature(enable = "simd128")]
#[doc(alias("i32x4.trunc_sat_f64x2_u_zero"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn u32x4_trunc_sat_f64x2_zero(a: v128) -> v128 {
    let ret: simd::u32x4 = unsafe {
        simd_shuffle!(
            simd_as::<simd::f64x2, simd::u32x2>(a.as_f64x2()),
            simd::u32x2::ZERO,
            [0, 1, 2, 3],
        )
    };
    ret.v128()
}

/// Lane-wise conversion from integer to floating point.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.convert_low_i32x4_s))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.convert_low_i32x4_s"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_convert_low_i32x4(a: v128) -> v128 {
    unsafe {
        simd_cast::<simd::i32x2, simd::f64x2>(simd_shuffle!(a.as_i32x4(), a.as_i32x4(), [0, 1],))
            .v128()
    }
}

/// Lane-wise conversion from integer to floating point.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.convert_low_i32x4_u))]
#[target_feature(enable = "simd128")]
#[doc(alias("f64x2.convert_low_i32x4_u"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_convert_low_u32x4(a: v128) -> v128 {
    unsafe {
        simd_cast::<simd::u32x2, simd::f64x2>(simd_shuffle!(a.as_u32x4(), a.as_u32x4(), [0, 1],))
            .v128()
    }
}

/// Conversion of the two double-precision floating point lanes to two lower
/// single-precision lanes of the result. The two higher lanes of the result are
/// initialized to zero. If the conversion result is not representable as a
/// single-precision floating point number, it is rounded to the nearest-even
/// representable number.
#[inline]
#[cfg_attr(test, assert_instr(f32x4.demote_f64x2_zero))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.demote_f64x2_zero"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f32x4_demote_f64x2_zero(a: v128) -> v128 {
    unsafe {
        simd_cast::<simd::f64x4, simd::f32x4>(simd_shuffle!(
            a.as_f64x2(),
            simd::f64x2::ZERO,
            [0, 1, 2, 3]
        ))
        .v128()
    }
}

/// Conversion of the two lower single-precision floating point lanes to the two
/// double-precision lanes of the result.
#[inline]
#[cfg_attr(test, assert_instr(f64x2.promote_low_f32x4))]
#[target_feature(enable = "simd128")]
#[doc(alias("f32x4.promote_low_f32x4"))]
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub fn f64x2_promote_low_f32x4(a: v128) -> v128 {
    unsafe {
        simd_cast::<simd::f32x2, simd::f64x2>(simd_shuffle!(a.as_f32x4(), a.as_f32x4(), [0, 1]))
            .v128()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::ops::{Add, Div, Mul, Neg, Sub};

    use std::fmt::Debug;
    use std::mem::transmute;
    use std::num::Wrapping;
    use std::prelude::v1::*;

    const _C1: v128 = i8x16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const _C2: v128 = u8x16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const _C3: v128 = i16x8(0, 1, 2, 3, 4, 5, 6, 7);
    const _C4: v128 = u16x8(0, 1, 2, 3, 4, 5, 6, 7);
    const _C5: v128 = i32x4(0, 1, 2, 3);
    const _C6: v128 = u32x4(0, 1, 2, 3);
    const _C7: v128 = i64x2(0, 1);
    const _C8: v128 = u64x2(0, 1);
    const _C9: v128 = f32x4(0.0, 1.0, 2.0, 3.0);
    const _C10: v128 = f64x2(0.0, 1.0);

    fn compare_bytes(a: v128, b: v128) {
        let a: [u8; 16] = unsafe { transmute(a) };
        let b: [u8; 16] = unsafe { transmute(b) };
        assert_eq!(a, b);
    }

    #[test]
    fn test_load() {
        unsafe {
            let arr: [i32; 4] = [0, 1, 2, 3];
            let vec = v128_load(arr.as_ptr() as *const v128);
            compare_bytes(vec, i32x4(0, 1, 2, 3));
        }
    }

    #[test]
    fn test_load_extend() {
        unsafe {
            let arr: [i8; 8] = [-3, -2, -1, 0, 1, 2, 3, 4];
            let vec = i16x8_load_extend_i8x8(arr.as_ptr());
            compare_bytes(vec, i16x8(-3, -2, -1, 0, 1, 2, 3, 4));
            let vec = i16x8_load_extend_u8x8(arr.as_ptr() as *const u8);
            compare_bytes(vec, i16x8(253, 254, 255, 0, 1, 2, 3, 4));

            let arr: [i16; 4] = [-1, 0, 1, 2];
            let vec = i32x4_load_extend_i16x4(arr.as_ptr());
            compare_bytes(vec, i32x4(-1, 0, 1, 2));
            let vec = i32x4_load_extend_u16x4(arr.as_ptr() as *const u16);
            compare_bytes(vec, i32x4(65535, 0, 1, 2));

            let arr: [i32; 2] = [-1, 1];
            let vec = i64x2_load_extend_i32x2(arr.as_ptr());
            compare_bytes(vec, i64x2(-1, 1));
            let vec = i64x2_load_extend_u32x2(arr.as_ptr() as *const u32);
            compare_bytes(vec, i64x2(u32::max_value().into(), 1));
        }
    }

    #[test]
    fn test_load_splat() {
        unsafe {
            compare_bytes(v128_load8_splat(&8), i8x16_splat(8));
            compare_bytes(v128_load16_splat(&9), i16x8_splat(9));
            compare_bytes(v128_load32_splat(&10), i32x4_splat(10));
            compare_bytes(v128_load64_splat(&11), i64x2_splat(11));
        }
    }

    #[test]
    fn test_load_zero() {
        unsafe {
            compare_bytes(v128_load32_zero(&10), i32x4(10, 0, 0, 0));
            compare_bytes(v128_load64_zero(&11), i64x2(11, 0));
        }
    }

    #[test]
    fn test_store() {
        unsafe {
            let mut spot = i8x16_splat(0);
            v128_store(&mut spot, i8x16_splat(1));
            compare_bytes(spot, i8x16_splat(1));
        }
    }

    #[test]
    fn test_load_lane() {
        unsafe {
            let zero = i8x16_splat(0);
            compare_bytes(
                v128_load8_lane::<2>(zero, &1),
                i8x16_replace_lane::<2>(zero, 1),
            );

            compare_bytes(
                v128_load16_lane::<2>(zero, &1),
                i16x8_replace_lane::<2>(zero, 1),
            );

            compare_bytes(
                v128_load32_lane::<2>(zero, &1),
                i32x4_replace_lane::<2>(zero, 1),
            );

            compare_bytes(
                v128_load64_lane::<1>(zero, &1),
                i64x2_replace_lane::<1>(zero, 1),
            );
        }
    }

    #[test]
    fn test_store_lane() {
        unsafe {
            let mut spot = 0;
            let zero = i8x16_splat(0);
            v128_store8_lane::<5>(i8x16_replace_lane::<5>(zero, 7), &mut spot);
            assert_eq!(spot, 7);

            let mut spot = 0;
            v128_store16_lane::<5>(i16x8_replace_lane::<5>(zero, 7), &mut spot);
            assert_eq!(spot, 7);

            let mut spot = 0;
            v128_store32_lane::<3>(i32x4_replace_lane::<3>(zero, 7), &mut spot);
            assert_eq!(spot, 7);

            let mut spot = 0;
            v128_store64_lane::<0>(i64x2_replace_lane::<0>(zero, 7), &mut spot);
            assert_eq!(spot, 7);
        }
    }

    #[test]
    fn test_i8x16() {
        const A: v128 = super::i8x16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        compare_bytes(A, A);

        const _: v128 = i16x8(0, 1, 2, 3, 4, 5, 6, 7);
        const _: v128 = i32x4(0, 1, 2, 3);
        const _: v128 = i64x2(0, 1);
        const _: v128 = f32x4(0., 1., 2., 3.);
        const _: v128 = f64x2(0., 1.);

        let bytes: [i16; 8] = unsafe { mem::transmute(i16x8(-1, -2, -3, -4, -5, -6, -7, -8)) };
        assert_eq!(bytes, [-1, -2, -3, -4, -5, -6, -7, -8]);
        let bytes: [i8; 16] = unsafe {
            mem::transmute(i8x16(
                -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
            ))
        };
        assert_eq!(
            bytes,
            [
                -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16
            ]
        );
    }

    #[test]
    fn test_shuffle() {
        let vec_a = i8x16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let vec_b = i8x16(
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        );

        let vec_r = i8x16_shuffle::<0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30>(
            vec_a, vec_b,
        );
        let vec_e = i8x16(0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30);
        compare_bytes(vec_r, vec_e);

        let vec_a = i16x8(0, 1, 2, 3, 4, 5, 6, 7);
        let vec_b = i16x8(8, 9, 10, 11, 12, 13, 14, 15);
        let vec_r = i16x8_shuffle::<0, 8, 2, 10, 4, 12, 6, 14>(vec_a, vec_b);
        let vec_e = i16x8(0, 8, 2, 10, 4, 12, 6, 14);
        compare_bytes(vec_r, vec_e);

        let vec_a = i32x4(0, 1, 2, 3);
        let vec_b = i32x4(4, 5, 6, 7);
        let vec_r = i32x4_shuffle::<0, 4, 2, 6>(vec_a, vec_b);
        let vec_e = i32x4(0, 4, 2, 6);
        compare_bytes(vec_r, vec_e);

        let vec_a = i64x2(0, 1);
        let vec_b = i64x2(2, 3);
        let vec_r = i64x2_shuffle::<0, 2>(vec_a, vec_b);
        let vec_e = i64x2(0, 2);
        compare_bytes(vec_r, vec_e);
    }

    // tests extract and replace lanes
    macro_rules! test_extract {
        (
            name: $test_id:ident,
            extract: $extract:ident,
            replace: $replace:ident,
            elem: $elem:ty,
            count: $count:expr,
            indices: [$($idx:expr),*],
        ) => {
            #[test]
            fn $test_id() {
                unsafe {
                    let arr: [$elem; $count] = [123 as $elem; $count];
                    let vec: v128 = transmute(arr);
                    $(
                        assert_eq!($extract::<$idx>(vec), 123 as $elem);
                    )*

                    // create a vector from array and check that the indices contain
                    // the same values as in the array:
                    let arr: [$elem; $count] = [$($idx as $elem),*];
                    let vec: v128 = transmute(arr);
                    $(
                        assert_eq!($extract::<$idx>(vec), $idx as $elem);

                        let tmp = $replace::<$idx>(vec, 124 as $elem);
                        assert_eq!($extract::<$idx>(tmp), 124 as $elem);
                    )*
                }
            }
        }
    }

    test_extract! {
        name: test_i8x16_extract_replace,
        extract: i8x16_extract_lane,
        replace: i8x16_replace_lane,
        elem: i8,
        count: 16,
        indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    }
    test_extract! {
        name: test_i16x8_extract_replace,
        extract: i16x8_extract_lane,
        replace: i16x8_replace_lane,
        elem: i16,
        count: 8,
        indices: [0, 1, 2, 3, 4, 5, 6, 7],
    }
    test_extract! {
        name: test_i32x4_extract_replace,
        extract: i32x4_extract_lane,
        replace: i32x4_replace_lane,
        elem: i32,
        count: 4,
        indices: [0, 1, 2, 3],
    }
    test_extract! {
        name: test_i64x2_extract_replace,
        extract: i64x2_extract_lane,
        replace: i64x2_replace_lane,
        elem: i64,
        count: 2,
        indices: [0, 1],
    }
    test_extract! {
        name: test_f32x4_extract_replace,
        extract: f32x4_extract_lane,
        replace: f32x4_replace_lane,
        elem: f32,
        count: 4,
        indices: [0, 1, 2, 3],
    }
    test_extract! {
        name: test_f64x2_extract_replace,
        extract: f64x2_extract_lane,
        replace: f64x2_replace_lane,
        elem: f64,
        count: 2,
        indices: [0, 1],
    }

    #[test]
    #[rustfmt::skip]
    fn test_swizzle() {
        compare_bytes(
            i8x16_swizzle(
                i32x4(1, 2, 3, 4),
                i8x16(
                    32, 31, 30, 29,
                    0, 1, 2, 3,
                    12, 13, 14, 15,
                    0, 4, 8, 12),
            ),
            i32x4(0, 1, 4, 0x04030201),
        );
    }

    macro_rules! test_splat {
        ($test_id:ident: $val:expr => $($vals:expr),*) => {
            #[test]
            fn $test_id() {
                let a = super::$test_id($val);
                let b = u8x16($($vals as u8),*);
                compare_bytes(a, b);
            }
        }
    }

    mod splats {
        use super::*;
        test_splat!(i8x16_splat: 42 => 42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42);
        test_splat!(i16x8_splat: 42 => 42, 0, 42, 0, 42, 0, 42, 0, 42, 0, 42, 0, 42, 0, 42, 0);
        test_splat!(i32x4_splat: 42 => 42, 0, 0, 0, 42, 0, 0, 0, 42, 0, 0, 0, 42, 0, 0, 0);
        test_splat!(i64x2_splat: 42 => 42, 0, 0, 0, 0, 0, 0, 0, 42, 0, 0, 0, 0, 0, 0, 0);
        test_splat!(f32x4_splat: 42. => 0, 0, 40, 66, 0, 0, 40, 66, 0, 0, 40, 66, 0, 0, 40, 66);
        test_splat!(f64x2_splat: 42. => 0, 0, 0, 0, 0, 0, 69, 64, 0, 0, 0, 0, 0, 0, 69, 64);
    }

    #[test]
    fn test_bitmasks() {
        let zero = i8x16_splat(0);
        let ones = i8x16_splat(!0);

        assert_eq!(i8x16_bitmask(zero), 0);
        assert_eq!(i8x16_bitmask(ones), 0xffff);
        assert_eq!(i8x16_bitmask(i8x16_splat(i8::MAX)), 0);
        assert_eq!(i8x16_bitmask(i8x16_splat(i8::MIN)), 0xffff);
        assert_eq!(i8x16_bitmask(i8x16_replace_lane::<1>(zero, -1)), 0b10);

        assert_eq!(i16x8_bitmask(zero), 0);
        assert_eq!(i16x8_bitmask(ones), 0xff);
        assert_eq!(i16x8_bitmask(i16x8_splat(i16::MAX)), 0);
        assert_eq!(i16x8_bitmask(i16x8_splat(i16::MIN)), 0xff);
        assert_eq!(i16x8_bitmask(i16x8_replace_lane::<1>(zero, -1)), 0b10);

        assert_eq!(i32x4_bitmask(zero), 0);
        assert_eq!(i32x4_bitmask(ones), 0b1111);
        assert_eq!(i32x4_bitmask(i32x4_splat(i32::MAX)), 0);
        assert_eq!(i32x4_bitmask(i32x4_splat(i32::MIN)), 0b1111);
        assert_eq!(i32x4_bitmask(i32x4_replace_lane::<1>(zero, -1)), 0b10);

        assert_eq!(i64x2_bitmask(zero), 0);
        assert_eq!(i64x2_bitmask(ones), 0b11);
        assert_eq!(i64x2_bitmask(i64x2_splat(i64::MAX)), 0);
        assert_eq!(i64x2_bitmask(i64x2_splat(i64::MIN)), 0b11);
        assert_eq!(i64x2_bitmask(i64x2_replace_lane::<1>(zero, -1)), 0b10);
    }

    #[test]
    fn test_narrow() {
        let zero = i8x16_splat(0);
        let ones = i8x16_splat(!0);

        compare_bytes(i8x16_narrow_i16x8(zero, zero), zero);
        compare_bytes(u8x16_narrow_i16x8(zero, zero), zero);
        compare_bytes(i8x16_narrow_i16x8(ones, ones), ones);
        compare_bytes(u8x16_narrow_i16x8(ones, ones), zero);

        compare_bytes(
            i8x16_narrow_i16x8(
                i16x8(
                    0,
                    1,
                    2,
                    -1,
                    i8::MIN.into(),
                    i8::MAX.into(),
                    u8::MIN.into(),
                    u8::MAX.into(),
                ),
                i16x8(
                    i16::MIN,
                    i16::MAX,
                    u16::MIN as i16,
                    u16::MAX as i16,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
            i8x16(0, 1, 2, -1, -128, 127, 0, 127, -128, 127, 0, -1, 0, 0, 0, 0),
        );

        compare_bytes(
            u8x16_narrow_i16x8(
                i16x8(
                    0,
                    1,
                    2,
                    -1,
                    i8::MIN.into(),
                    i8::MAX.into(),
                    u8::MIN.into(),
                    u8::MAX.into(),
                ),
                i16x8(
                    i16::MIN,
                    i16::MAX,
                    u16::MIN as i16,
                    u16::MAX as i16,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
            i8x16(0, 1, 2, 0, 0, 127, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0),
        );

        compare_bytes(i16x8_narrow_i32x4(zero, zero), zero);
        compare_bytes(u16x8_narrow_i32x4(zero, zero), zero);
        compare_bytes(i16x8_narrow_i32x4(ones, ones), ones);
        compare_bytes(u16x8_narrow_i32x4(ones, ones), zero);

        compare_bytes(
            i16x8_narrow_i32x4(
                i32x4(0, -1, i16::MIN.into(), i16::MAX.into()),
                i32x4(i32::MIN, i32::MAX, u32::MIN as i32, u32::MAX as i32),
            ),
            i16x8(0, -1, i16::MIN, i16::MAX, i16::MIN, i16::MAX, 0, -1),
        );

        compare_bytes(
            u16x8_narrow_i32x4(
                i32x4(u16::MAX.into(), -1, i16::MIN.into(), i16::MAX.into()),
                i32x4(i32::MIN, i32::MAX, u32::MIN as i32, u32::MAX as i32),
            ),
            i16x8(-1, 0, 0, i16::MAX, 0, -1, 0, 0),
        );
    }

    #[test]
    fn test_extend() {
        let zero = i8x16_splat(0);
        let ones = i8x16_splat(!0);

        compare_bytes(i16x8_extend_low_i8x16(zero), zero);
        compare_bytes(i16x8_extend_high_i8x16(zero), zero);
        compare_bytes(i16x8_extend_low_u8x16(zero), zero);
        compare_bytes(i16x8_extend_high_u8x16(zero), zero);
        compare_bytes(i16x8_extend_low_i8x16(ones), ones);
        compare_bytes(i16x8_extend_high_i8x16(ones), ones);
        let halves = u16x8_splat(u8::MAX.into());
        compare_bytes(i16x8_extend_low_u8x16(ones), halves);
        compare_bytes(i16x8_extend_high_u8x16(ones), halves);

        compare_bytes(i32x4_extend_low_i16x8(zero), zero);
        compare_bytes(i32x4_extend_high_i16x8(zero), zero);
        compare_bytes(i32x4_extend_low_u16x8(zero), zero);
        compare_bytes(i32x4_extend_high_u16x8(zero), zero);
        compare_bytes(i32x4_extend_low_i16x8(ones), ones);
        compare_bytes(i32x4_extend_high_i16x8(ones), ones);
        let halves = u32x4_splat(u16::MAX.into());
        compare_bytes(i32x4_extend_low_u16x8(ones), halves);
        compare_bytes(i32x4_extend_high_u16x8(ones), halves);

        compare_bytes(i64x2_extend_low_i32x4(zero), zero);
        compare_bytes(i64x2_extend_high_i32x4(zero), zero);
        compare_bytes(i64x2_extend_low_u32x4(zero), zero);
        compare_bytes(i64x2_extend_high_u32x4(zero), zero);
        compare_bytes(i64x2_extend_low_i32x4(ones), ones);
        compare_bytes(i64x2_extend_high_i32x4(ones), ones);
        let halves = i64x2_splat(u32::MAX.into());
        compare_bytes(u64x2_extend_low_u32x4(ones), halves);
        compare_bytes(u64x2_extend_high_u32x4(ones), halves);
    }

    #[test]
    fn test_dot() {
        let zero = i8x16_splat(0);
        let ones = i8x16_splat(!0);
        let two = i32x4_splat(2);
        compare_bytes(i32x4_dot_i16x8(zero, zero), zero);
        compare_bytes(i32x4_dot_i16x8(ones, ones), two);
    }

    macro_rules! test_binop {
        (
            $($name:ident => {
                $([$($vec1:tt)*] ($op:ident | $f:ident) [$($vec2:tt)*],)*
            })*
        ) => ($(
            #[test]
            fn $name() {
                unsafe {
                    $(
                        let v1 = [$($vec1)*];
                        let v2 = [$($vec2)*];
                        let v1_v128: v128 = mem::transmute(v1);
                        let v2_v128: v128 = mem::transmute(v2);
                        let v3_v128 = super::$f(v1_v128, v2_v128);
                        let mut v3 = [$($vec1)*];
                        let _ignore = v3;
                        v3 = mem::transmute(v3_v128);

                        for (i, actual) in v3.iter().enumerate() {
                            let expected = v1[i].$op(v2[i]);
                            assert_eq!(*actual, expected);
                        }
                    )*
                }
            }
        )*)
    }

    macro_rules! test_unop {
        (
            $($name:ident => {
                $(($op:ident | $f:ident) [$($vec1:tt)*],)*
            })*
        ) => ($(
            #[test]
            fn $name() {
                unsafe {
                    $(
                        let v1 = [$($vec1)*];
                        let v1_v128: v128 = mem::transmute(v1);
                        let v2_v128 = super::$f(v1_v128);
                        let mut v2 = [$($vec1)*];
                        let _ignore = v2;
                        v2 = mem::transmute(v2_v128);

                        for (i, actual) in v2.iter().enumerate() {
                            let expected = v1[i].$op();
                            assert_eq!(*actual, expected);
                        }
                    )*
                }
            }
        )*)
    }

    trait Avgr: Sized {
        fn avgr(self, other: Self) -> Self;
    }

    macro_rules! impl_avgr {
        ($($i:ident)*) => ($(impl Avgr for $i {
            fn avgr(self, other: Self) -> Self {
                ((self as u64 + other as u64 + 1) / 2) as $i
            }
        })*)
    }

    impl_avgr!(u8 u16);

    test_binop! {
        test_i8x16_add => {
            [0i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                (wrapping_add | i8x16_add)
            [1i8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

            [1i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (wrapping_add | i8x16_add)
            [-2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -18],

            [1i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (wrapping_add | i8x16_add)
            [127, -44, 43, 126, 4, 2, 9, -3, -59, -43, 39, -69, 79, -3, 9, -24],
        }

        test_i8x16_add_sat_s => {
            [0i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                (saturating_add | i8x16_add_sat)
            [1i8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

            [1i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (saturating_add | i8x16_add_sat)
            [-2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -18],

            [1i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (saturating_add | i8x16_add_sat)
            [127, -44, 43, 126, 4, 2, 9, -3, -59, -43, 39, -69, 79, -3, 9, -24],
        }

        test_i8x16_add_sat_u => {
            [0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                (saturating_add | u8x16_add_sat)
            [1u8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

            [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (saturating_add | u8x16_add_sat)
            [255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240],

            [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (saturating_add | u8x16_add_sat)
            [127, -44i8 as u8, 43, 126, 4, 2, 9, -3i8 as u8, -59i8 as u8, -43i8 as u8, 39, -69i8 as u8, 79, -3i8 as u8, 9, -24i8 as u8],
        }

        test_i8x16_sub => {
            [0i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                (wrapping_sub | i8x16_sub)
            [1i8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

            [1i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (wrapping_sub | i8x16_sub)
            [-2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -18],

            [1i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (wrapping_sub | i8x16_sub)
            [-127, -44, 43, 126, 4, 2, 9, -3, -59, -43, 39, -69, 79, -3, 4, 8],
        }

        test_i8x16_sub_sat_s => {
            [0i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                (saturating_sub | i8x16_sub_sat)
            [1i8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

            [1i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (saturating_sub | i8x16_sub_sat)
            [-2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -18],

            [1i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (saturating_sub | i8x16_sub_sat)
            [-127, -44, 43, 126, 4, 2, 9, -3, -59, -43, 39, -69, 79, -3, 4, 8],
        }

        test_i8x16_sub_sat_u => {
            [0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                (saturating_sub | u8x16_sub_sat)
            [1u8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

            [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (saturating_sub | u8x16_sub_sat)
            [255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240],

            [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (saturating_sub | u8x16_sub_sat)
            [127, -44i8 as u8, 43, 126, 4, 2, 9, -3i8 as u8, -59i8 as u8, -43i8 as u8, 39, -69i8 as u8, 79, -3i8 as u8, 9, -24i8 as u8],
        }

        test_i8x16_min_s => {
            [0i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                (min | i8x16_min)
            [1i8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

            [1i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (min | i8x16_min)
            [-2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -18],

            [1i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (min | i8x16_min)
            [-127, -44, 43, 126, 4, 2, 9, -3, -59, -43, 39, -69, 79, -3, 4, 8],
        }

        test_i8x16_min_u => {
            [0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                (min | u8x16_min)
            [1u8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

            [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (min | u8x16_min)
            [255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240],

            [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (min | u8x16_min)
            [127, -44i8 as u8, 43, 126, 4, 2, 9, -3i8 as u8, -59i8 as u8, -43i8 as u8, 39, -69i8 as u8, 79, -3i8 as u8, 9, -24i8 as u8],
        }

        test_i8x16_max_s => {
            [0i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                (max | i8x16_max)
            [1i8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

            [1i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (max | i8x16_max)
            [-2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -18],

            [1i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (max | i8x16_max)
            [-127, -44, 43, 126, 4, 2, 9, -3, -59, -43, 39, -69, 79, -3, 4, 8],
        }

        test_i8x16_max_u => {
            [0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                (max | u8x16_max)
            [1u8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

            [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (max | u8x16_max)
            [255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240],

            [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (max | u8x16_max)
            [127, -44i8 as u8, 43, 126, 4, 2, 9, -3i8 as u8, -59i8 as u8, -43i8 as u8, 39, -69i8 as u8, 79, -3i8 as u8, 9, -24i8 as u8],
        }

        test_i8x16_avgr_u => {
            [0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                (avgr | u8x16_avgr)
            [1u8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

            [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (avgr | u8x16_avgr)
            [255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240],

            [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                (avgr | u8x16_avgr)
            [127, -44i8 as u8, 43, 126, 4, 2, 9, -3i8 as u8, -59i8 as u8, -43i8 as u8, 39, -69i8 as u8, 79, -3i8 as u8, 9, -24i8 as u8],
        }

        test_i16x8_add => {
            [0i16, 0, 0, 0, 0, 0, 0, 0]
                (wrapping_add | i16x8_add)
            [1i16, 1, 1, 1, 1, 1, 1, 1],

            [1i16, 2, 3, 4, 5, 6, 7, 8]
                (wrapping_add | i16x8_add)
            [32767, 8, -2494,-4, 4882, -4, 848, 3830],
        }

        test_i16x8_add_sat_s => {
            [0i16, 0, 0, 0, 0, 0, 0, 0]
                (saturating_add | i16x8_add_sat)
            [1i16, 1, 1, 1, 1, 1, 1, 1],

            [1i16, 2, 3, 4, 5, 6, 7, 8]
                (saturating_add | i16x8_add_sat)
            [32767, 8, -2494,-4, 4882, -4, 848, 3830],
        }

        test_i16x8_add_sat_u => {
            [0u16, 0, 0, 0, 0, 0, 0, 0]
                (saturating_add | u16x8_add_sat)
            [1u16, 1, 1, 1, 1, 1, 1, 1],

            [1u16, 2, 3, 4, 5, 6, 7, 8]
                (saturating_add | u16x8_add_sat)
            [32767, 8, -2494i16 as u16,-4i16 as u16, 4882, -4i16 as u16, 848, 3830],
        }

        test_i16x8_sub => {
            [0i16, 0, 0, 0, 0, 0, 0, 0]
                (wrapping_sub | i16x8_sub)
            [1i16, 1, 1, 1, 1, 1, 1, 1],

            [1i16, 2, 3, 4, 5, 6, 7, 8]
                (wrapping_sub | i16x8_sub)
            [32767, 8, -2494,-4, 4882, -4, 848, 3830],
        }

        test_i16x8_sub_sat_s => {
            [0i16, 0, 0, 0, 0, 0, 0, 0]
                (saturating_sub | i16x8_sub_sat)
            [1i16, 1, 1, 1, 1, 1, 1, 1],

            [1i16, 2, 3, 4, 5, 6, 7, 8]
                (saturating_sub | i16x8_sub_sat)
            [32767, 8, -2494,-4, 4882, -4, 848, 3830],
        }

        test_i16x8_sub_sat_u => {
            [0u16, 0, 0, 0, 0, 0, 0, 0]
                (saturating_sub | u16x8_sub_sat)
            [1u16, 1, 1, 1, 1, 1, 1, 1],

            [1u16, 2, 3, 4, 5, 6, 7, 8]
                (saturating_sub | u16x8_sub_sat)
            [32767, 8, -2494i16 as u16,-4i16 as u16, 4882, -4i16 as u16, 848, 3830],
        }

        test_i16x8_mul => {
            [0i16, 0, 0, 0, 0, 0, 0, 0]
                (wrapping_mul | i16x8_mul)
            [1i16, 1, 1, 1, 1, 1, 1, 1],

            [1i16, 2, 3, 4, 5, 6, 7, 8]
                (wrapping_mul | i16x8_mul)
            [32767, 8, -2494,-4, 4882, -4, 848, 3830],
        }

        test_i16x8_min_s => {
            [0i16, 0, 0, 0, 0, 0, 0, 0]
                (min | i16x8_min)
            [1i16, 1, 1, 1, 1, 1, 1, 1],

            [1i16, 2, 3, 4, 5, 6, 7, 8]
                (min | i16x8_min)
            [32767, 8, -2494,-4, 4882, -4, 848, 3830],
        }

        test_i16x8_min_u => {
            [0u16, 0, 0, 0, 0, 0, 0, 0]
                (min | u16x8_min)
            [1u16, 1, 1, 1, 1, 1, 1, 1],

            [1u16, 2, 3, 4, 5, 6, 7, 8]
                (min | u16x8_min)
            [32767, 8, -2494i16 as u16,-4i16 as u16, 4882, -4i16 as u16, 848, 3830],
        }

        test_i16x8_max_s => {
            [0i16, 0, 0, 0, 0, 0, 0, 0]
                (max | i16x8_max)
            [1i16, 1, 1, 1, 1, 1, 1, 1],

            [1i16, 2, 3, 4, 5, 6, 7, 8]
                (max | i16x8_max)
            [32767, 8, -2494,-4, 4882, -4, 848, 3830],
        }

        test_i16x8_max_u => {
            [0u16, 0, 0, 0, 0, 0, 0, 0]
                (max | u16x8_max)
            [1u16, 1, 1, 1, 1, 1, 1, 1],

            [1u16, 2, 3, 4, 5, 6, 7, 8]
                (max | u16x8_max)
            [32767, 8, -2494i16 as u16,-4i16 as u16, 4882, -4i16 as u16, 848, 3830],
        }

        test_i16x8_avgr_u => {
            [0u16, 0, 0, 0, 0, 0, 0, 0]
                (avgr | u16x8_avgr)
            [1u16, 1, 1, 1, 1, 1, 1, 1],

            [1u16, 2, 3, 4, 5, 6, 7, 8]
                (avgr | u16x8_avgr)
            [32767, 8, -2494i16 as u16,-4i16 as u16, 4882, -4i16 as u16, 848, 3830],
        }

        test_i32x4_add => {
            [0i32, 0, 0, 0] (wrapping_add | i32x4_add) [1, 2, 3, 4],
            [1i32, 1283, i32::MAX, i32::MIN]
                (wrapping_add | i32x4_add)
            [i32::MAX; 4],
        }

        test_i32x4_sub => {
            [0i32, 0, 0, 0] (wrapping_sub | i32x4_sub) [1, 2, 3, 4],
            [1i32, 1283, i32::MAX, i32::MIN]
                (wrapping_sub | i32x4_sub)
            [i32::MAX; 4],
        }

        test_i32x4_mul => {
            [0i32, 0, 0, 0] (wrapping_mul | i32x4_mul) [1, 2, 3, 4],
            [1i32, 1283, i32::MAX, i32::MIN]
                (wrapping_mul | i32x4_mul)
            [i32::MAX; 4],
        }

        test_i32x4_min_s => {
            [0i32, 0, 0, 0] (min | i32x4_min) [1, 2, 3, 4],
            [1i32, 1283, i32::MAX, i32::MIN]
                (min | i32x4_min)
            [i32::MAX; 4],
        }

        test_i32x4_min_u => {
            [0u32, 0, 0, 0] (min | u32x4_min) [1, 2, 3, 4],
            [1u32, 1283, i32::MAX as u32, i32::MIN as u32]
                (min | u32x4_min)
            [i32::MAX as u32; 4],
        }

        test_i32x4_max_s => {
            [0i32, 0, 0, 0] (max | i32x4_max) [1, 2, 3, 4],
            [1i32, 1283, i32::MAX, i32::MIN]
                (max | i32x4_max)
            [i32::MAX; 4],
        }

        test_i32x4_max_u => {
            [0u32, 0, 0, 0] (max | u32x4_max) [1, 2, 3, 4],
            [1u32, 1283, i32::MAX as u32, i32::MIN as u32]
                (max | u32x4_max)
            [i32::MAX as u32; 4],
        }

        test_i64x2_add => {
            [0i64, 0] (wrapping_add | i64x2_add) [1, 2],
            [i64::MIN, i64::MAX] (wrapping_add | i64x2_add) [i64::MAX, i64::MIN],
            [i64::MAX; 2] (wrapping_add | i64x2_add) [i64::MAX; 2],
            [-4i64, -4] (wrapping_add | i64x2_add) [800, 939],
        }

        test_i64x2_sub => {
            [0i64, 0] (wrapping_sub | i64x2_sub) [1, 2],
            [i64::MIN, i64::MAX] (wrapping_sub | i64x2_sub) [i64::MAX, i64::MIN],
            [i64::MAX; 2] (wrapping_sub | i64x2_sub) [i64::MAX; 2],
            [-4i64, -4] (wrapping_sub | i64x2_sub) [800, 939],
        }

        test_i64x2_mul => {
            [0i64, 0] (wrapping_mul | i64x2_mul) [1, 2],
            [i64::MIN, i64::MAX] (wrapping_mul | i64x2_mul) [i64::MAX, i64::MIN],
            [i64::MAX; 2] (wrapping_mul | i64x2_mul) [i64::MAX; 2],
            [-4i64, -4] (wrapping_mul | i64x2_mul) [800, 939],
        }

        test_f32x4_add => {
            [-1.0f32, 2.0, 3.0, 4.0] (add | f32x4_add) [1., 2., 0., 0.],
            [f32::INFINITY, -0.0, f32::NEG_INFINITY, 3.0]
                (add | f32x4_add)
            [1., 2., 0., 0.],
        }

        test_f32x4_sub => {
            [-1.0f32, 2.0, 3.0, 4.0] (sub | f32x4_sub) [1., 2., 0., 0.],
            [f32::INFINITY, -0.0, f32::NEG_INFINITY, 3.0]
                (sub | f32x4_sub)
            [1., 2., 0., 0.],
        }

        test_f32x4_mul => {
            [-1.0f32, 2.0, 3.0, 4.0] (mul | f32x4_mul) [1., 2., 0., 0.],
            [f32::INFINITY, -0.0, f32::NEG_INFINITY, 3.0]
                (mul | f32x4_mul)
            [1., 2., 1., 0.],
        }

        test_f32x4_div => {
            [-1.0f32, 2.0, 3.0, 4.0] (div | f32x4_div) [1., 2., 0., 0.],
            [f32::INFINITY, -0.0, f32::NEG_INFINITY, 3.0]
                (div | f32x4_div)
            [1., 2., 0., 0.],
        }

        test_f32x4_min => {
            [-1.0f32, 2.0, 3.0, 4.0] (min | f32x4_min) [1., 2., 0., 0.],
            [f32::INFINITY, -0.0, f32::NEG_INFINITY, 3.0]
                (min | f32x4_min)
            [1., 2., 0., 0.],
        }

        test_f32x4_max => {
            [-1.0f32, 2.0, 3.0, 4.0] (max | f32x4_max) [1., 2., 0., 0.],
            [f32::INFINITY, -0.0, f32::NEG_INFINITY, 3.0]
                (max | f32x4_max)
            [1., 2., 0., 0.],
        }

        test_f32x4_pmin => {
            [-1.0f32, 2.0, 3.0, 4.0] (min | f32x4_pmin) [1., 2., 0., 0.],
            [f32::INFINITY, -0.0, f32::NEG_INFINITY, 3.0]
                (min | f32x4_pmin)
            [1., 2., 0., 0.],
        }

        test_f32x4_pmax => {
            [-1.0f32, 2.0, 3.0, 4.0] (max | f32x4_pmax) [1., 2., 0., 0.],
            [f32::INFINITY, -0.0, f32::NEG_INFINITY, 3.0]
                (max | f32x4_pmax)
            [1., 2., 0., 0.],
        }

        test_f64x2_add => {
            [-1.0f64, 2.0] (add | f64x2_add) [1., 2.],
            [f64::INFINITY, f64::NEG_INFINITY] (add | f64x2_add) [1., 2.],
        }

        test_f64x2_sub => {
            [-1.0f64, 2.0] (sub | f64x2_sub) [1., 2.],
            [f64::INFINITY, f64::NEG_INFINITY] (sub | f64x2_sub) [1., 2.],
        }

        test_f64x2_mul => {
            [-1.0f64, 2.0] (mul | f64x2_mul) [1., 2.],
            [f64::INFINITY, f64::NEG_INFINITY] (mul | f64x2_mul) [1., 2.],
        }

        test_f64x2_div => {
            [-1.0f64, 2.0] (div | f64x2_div) [1., 2.],
            [f64::INFINITY, f64::NEG_INFINITY] (div | f64x2_div) [1., 2.],
        }

        test_f64x2_min => {
            [-1.0f64, 2.0] (min | f64x2_min) [1., 2.],
            [f64::INFINITY, f64::NEG_INFINITY] (min | f64x2_min) [1., 2.],
        }

        test_f64x2_max => {
            [-1.0f64, 2.0] (max | f64x2_max) [1., 2.],
            [f64::INFINITY, f64::NEG_INFINITY] (max | f64x2_max) [1., 2.],
        }

        test_f64x2_pmin => {
            [-1.0f64, 2.0] (min | f64x2_pmin) [1., 2.],
            [f64::INFINITY, f64::NEG_INFINITY] (min | f64x2_pmin) [1., 2.],
        }

        test_f64x2_pmax => {
            [-1.0f64, 2.0] (max | f64x2_pmax) [1., 2.],
            [f64::INFINITY, f64::NEG_INFINITY] (max | f64x2_pmax) [1., 2.],
        }
    }

    test_unop! {
        test_i8x16_abs => {
            (wrapping_abs | i8x16_abs)
            [1i8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

            (wrapping_abs | i8x16_abs)
            [-2i8, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -18],

            (wrapping_abs | i8x16_abs)
            [-127i8, -44, 43, 126, 4, -128, 127, -59, -43, 39, -69, 79, -3, 35, 83, 13],
        }

        test_i8x16_neg => {
            (wrapping_neg | i8x16_neg)
            [1i8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

            (wrapping_neg | i8x16_neg)
            [-2i8, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -18],

            (wrapping_neg | i8x16_neg)
            [-127i8, -44, 43, 126, 4, -128, 127, -59, -43, 39, -69, 79, -3, 35, 83, 13],
        }

        test_i16x8_abs => {
            (wrapping_abs | i16x8_abs) [1i16, 1, 1, 1, 1, 1, 1, 1],
            (wrapping_abs | i16x8_abs) [2i16, 0x7fff, !0, 4, 42, -5, 33, -4847],
        }

        test_i16x8_neg => {
            (wrapping_neg | i16x8_neg) [1i16, 1, 1, 1, 1, 1, 1, 1],
            (wrapping_neg | i16x8_neg) [2i16, 0x7fff, !0, 4, 42, -5, 33, -4847],
        }

        test_i32x4_abs => {
            (wrapping_abs | i32x4_abs) [1i32, 2, 3, 4],
            (wrapping_abs | i32x4_abs) [i32::MIN, i32::MAX, 0, 4],
        }

        test_i32x4_neg => {
            (wrapping_neg | i32x4_neg) [1i32, 2, 3, 4],
            (wrapping_neg | i32x4_neg) [i32::MIN, i32::MAX, 0, 4],
        }

        test_i64x2_abs => {
            (wrapping_abs | i64x2_abs) [1i64, 2],
            (wrapping_abs | i64x2_abs) [i64::MIN, i64::MAX],
        }

        test_i64x2_neg => {
            (wrapping_neg | i64x2_neg) [1i64, 2],
            (wrapping_neg | i64x2_neg) [i64::MIN, i64::MAX],
        }

        test_f32x4_ceil => {
            (ceil | f32x4_ceil) [1.0f32, 2., 2.5, 3.3],
            (ceil | f32x4_ceil) [0.0, -0.3, f32::INFINITY, -0.0],
        }

        test_f32x4_floor => {
            (floor | f32x4_floor) [1.0f32, 2., 2.5, 3.3],
            (floor | f32x4_floor) [0.0, -0.3, f32::INFINITY, -0.0],
        }

        test_f32x4_trunc => {
            (trunc | f32x4_trunc) [1.0f32, 2., 2.5, 3.3],
            (trunc | f32x4_trunc) [0.0, -0.3, f32::INFINITY, -0.0],
        }

        test_f32x4_nearest => {
            (round | f32x4_nearest) [1.0f32, 2., 2.6, 3.3],
            (round | f32x4_nearest) [0.0, -0.3, f32::INFINITY, -0.0],
        }

        test_f32x4_abs => {
            (abs | f32x4_abs) [1.0f32, 2., 2.6, 3.3],
            (abs | f32x4_abs) [0.0, -0.3, f32::INFINITY, -0.0],
        }

        test_f32x4_neg => {
            (neg | f32x4_neg) [1.0f32, 2., 2.6, 3.3],
            (neg | f32x4_neg) [0.0, -0.3, f32::INFINITY, -0.0],
        }

        test_f32x4_sqrt => {
            (sqrt | f32x4_sqrt) [1.0f32, 2., 2.6, 3.3],
            (sqrt | f32x4_sqrt) [0.0, 0.3, f32::INFINITY, 0.1],
        }

        test_f64x2_ceil => {
            (ceil | f64x2_ceil) [1.0f64, 2.3],
            (ceil | f64x2_ceil) [f64::INFINITY, -0.1],
        }

        test_f64x2_floor => {
            (floor | f64x2_floor) [1.0f64, 2.3],
            (floor | f64x2_floor) [f64::INFINITY, -0.1],
        }

        test_f64x2_trunc => {
            (trunc | f64x2_trunc) [1.0f64, 2.3],
            (trunc | f64x2_trunc) [f64::INFINITY, -0.1],
        }

        test_f64x2_nearest => {
            (round | f64x2_nearest) [1.0f64, 2.3],
            (round | f64x2_nearest) [f64::INFINITY, -0.1],
        }

        test_f64x2_abs => {
            (abs | f64x2_abs) [1.0f64, 2.3],
            (abs | f64x2_abs) [f64::INFINITY, -0.1],
        }

        test_f64x2_neg => {
            (neg | f64x2_neg) [1.0f64, 2.3],
            (neg | f64x2_neg) [f64::INFINITY, -0.1],
        }

        test_f64x2_sqrt => {
            (sqrt | f64x2_sqrt) [1.0f64, 2.3],
            (sqrt | f64x2_sqrt) [f64::INFINITY, 0.1],
        }
    }

    macro_rules! floating_point {
        (f32) => {
            true
        };
        (f64) => {
            true
        };
        ($id:ident) => {
            false
        };
    }

    trait IsNan: Sized {
        fn is_nan(self) -> bool {
            false
        }
    }
    impl IsNan for i8 {}
    impl IsNan for i16 {}
    impl IsNan for i32 {}
    impl IsNan for i64 {}

    macro_rules! test_bop {
         ($id:ident[$ety:ident; $ecount:expr] |
          $binary_op:ident [$op_test_id:ident] :
          ([$($in_a:expr),*], [$($in_b:expr),*]) => [$($out:expr),*]) => {
             test_bop!(
                 $id[$ety; $ecount] => $ety | $binary_op [ $op_test_id ]:
                 ([$($in_a),*], [$($in_b),*]) => [$($out),*]
             );

         };
         ($id:ident[$ety:ident; $ecount:expr] => $oty:ident |
          $binary_op:ident [$op_test_id:ident] :
          ([$($in_a:expr),*], [$($in_b:expr),*]) => [$($out:expr),*]) => {
             #[test]
             fn $op_test_id() {
                 unsafe {
                     let a_input: [$ety; $ecount] = [$($in_a),*];
                     let b_input: [$ety; $ecount] = [$($in_b),*];
                     let output: [$oty; $ecount] = [$($out),*];

                     let a_vec_in: v128 = transmute(a_input);
                     let b_vec_in: v128 = transmute(b_input);
                     let vec_res: v128 = $binary_op(a_vec_in, b_vec_in);

                     let res: [$oty; $ecount] = transmute(vec_res);

                     if !floating_point!($ety) {
                         assert_eq!(res, output);
                     } else {
                         for i in 0..$ecount {
                             let r = res[i];
                             let o = output[i];
                             assert_eq!(r.is_nan(), o.is_nan());
                             if !r.is_nan() {
                                 assert_eq!(r, o);
                             }
                         }
                     }
                 }
             }
         }
     }

    macro_rules! test_bops {
         ($id:ident[$ety:ident; $ecount:expr] |
          $binary_op:ident [$op_test_id:ident]:
          ([$($in_a:expr),*], $in_b:expr) => [$($out:expr),*]) => {
             #[test]
             fn $op_test_id() {
                 unsafe {
                     let a_input: [$ety; $ecount] = [$($in_a),*];
                     let output: [$ety; $ecount] = [$($out),*];

                     let a_vec_in: v128 = transmute(a_input);
                     let vec_res: v128 = $binary_op(a_vec_in, $in_b);

                     let res: [$ety; $ecount] = transmute(vec_res);
                     assert_eq!(res, output);
                 }
             }
         }
     }

    macro_rules! test_uop {
         ($id:ident[$ety:ident; $ecount:expr] |
          $unary_op:ident [$op_test_id:ident]: [$($in_a:expr),*] => [$($out:expr),*]) => {
             #[test]
             fn $op_test_id() {
                 unsafe {
                     let a_input: [$ety; $ecount] = [$($in_a),*];
                     let output: [$ety; $ecount] = [$($out),*];

                     let a_vec_in: v128 = transmute(a_input);
                     let vec_res: v128 = $unary_op(a_vec_in);

                     let res: [$ety; $ecount] = transmute(vec_res);
                     assert_eq!(res, output);
                 }
             }
         }
     }

    test_bops!(i8x16[i8; 16] | i8x16_shl[i8x16_shl_test]:
               ([0, -1, 2, 3, 4, 5, 6, i8::MAX, 1, 1, 1, 1, 1, 1, 1, 1], 1) =>
               [0, -2, 4, 6, 8, 10, 12, -2, 2, 2, 2, 2, 2, 2, 2, 2]);
    test_bops!(i16x8[i16; 8] | i16x8_shl[i16x8_shl_test]:
                ([0, -1, 2, 3, 4, 5, 6, i16::MAX], 1) =>
                [0, -2, 4, 6, 8, 10, 12, -2]);
    test_bops!(i32x4[i32; 4] | i32x4_shl[i32x4_shl_test]:
                ([0, -1, 2, 3], 1) => [0, -2, 4, 6]);
    test_bops!(i64x2[i64; 2] | i64x2_shl[i64x2_shl_test]:
                ([0, -1], 1) => [0, -2]);

    test_bops!(i8x16[i8; 16] | i8x16_shr[i8x16_shr_s_test]:
               ([0, -1, 2, 3, 4, 5, 6, i8::MAX, 1, 1, 1, 1, 1, 1, 1, 1], 1) =>
               [0, -1, 1, 1, 2, 2, 3, 63, 0, 0, 0, 0, 0, 0, 0, 0]);
    test_bops!(i16x8[i16; 8] | i16x8_shr[i16x8_shr_s_test]:
               ([0, -1, 2, 3, 4, 5, 6, i16::MAX], 1) =>
               [0, -1, 1, 1, 2, 2, 3, i16::MAX / 2]);
    test_bops!(i32x4[i32; 4] | i32x4_shr[i32x4_shr_s_test]:
               ([0, -1, 2, 3], 1) => [0, -1, 1, 1]);
    test_bops!(i64x2[i64; 2] | i64x2_shr[i64x2_shr_s_test]:
               ([0, -1], 1) => [0, -1]);

    test_bops!(i8x16[i8; 16] | u8x16_shr[i8x16_uhr_u_test]:
                ([0, -1, 2, 3, 4, 5, 6, i8::MAX, 1, 1, 1, 1, 1, 1, 1, 1], 1) =>
                [0, i8::MAX, 1, 1, 2, 2, 3, 63, 0, 0, 0, 0, 0, 0, 0, 0]);
    test_bops!(i16x8[i16; 8] | u16x8_shr[i16x8_uhr_u_test]:
                ([0, -1, 2, 3, 4, 5, 6, i16::MAX], 1) =>
                [0, i16::MAX, 1, 1, 2, 2, 3, i16::MAX / 2]);
    test_bops!(i32x4[i32; 4] | u32x4_shr[i32x4_uhr_u_test]:
                ([0, -1, 2, 3], 1) => [0, i32::MAX, 1, 1]);
    test_bops!(i64x2[i64; 2] | u64x2_shr[i64x2_uhr_u_test]:
                ([0, -1], 1) => [0, i64::MAX]);

    #[test]
    fn v128_bitwise_logical_ops() {
        unsafe {
            let a: [u32; 4] = [u32::MAX, 0, u32::MAX, 0];
            let b: [u32; 4] = [u32::MAX; 4];
            let c: [u32; 4] = [0; 4];

            let vec_a: v128 = transmute(a);
            let vec_b: v128 = transmute(b);
            let vec_c: v128 = transmute(c);

            let r: v128 = v128_and(vec_a, vec_a);
            compare_bytes(r, vec_a);
            let r: v128 = v128_and(vec_a, vec_b);
            compare_bytes(r, vec_a);
            let r: v128 = v128_andnot(vec_a, vec_b);
            compare_bytes(r, vec_c);
            let r: v128 = v128_andnot(vec_a, vec_a);
            compare_bytes(r, vec_c);
            let r: v128 = v128_andnot(vec_a, vec_c);
            compare_bytes(r, vec_a);
            let r: v128 = v128_or(vec_a, vec_b);
            compare_bytes(r, vec_b);
            let r: v128 = v128_not(vec_b);
            compare_bytes(r, vec_c);
            let r: v128 = v128_xor(vec_a, vec_c);
            compare_bytes(r, vec_a);

            let r: v128 = v128_bitselect(vec_b, vec_c, vec_b);
            compare_bytes(r, vec_b);
            let r: v128 = v128_bitselect(vec_b, vec_c, vec_c);
            compare_bytes(r, vec_c);
            let r: v128 = v128_bitselect(vec_b, vec_c, vec_a);
            compare_bytes(r, vec_a);
        }
    }

    macro_rules! test_bool_red {
         ([$test_id:ident, $any:ident, $all:ident] | [$($true:expr),*] | [$($false:expr),*] | [$($alt:expr),*]) => {
             #[test]
             fn $test_id() {
                 unsafe {
                     let vec_a: v128 = transmute([$($true),*]); // true
                     let vec_b: v128 = transmute([$($false),*]); // false
                     let vec_c: v128 = transmute([$($alt),*]); // alternating

                     // TODO
                     // assert_eq!($any(vec_a), true);
                     // assert_eq!($any(vec_b), false);
                     // assert_eq!($any(vec_c), true);

                     assert_eq!($all(vec_a), true);
                     assert_eq!($all(vec_b), false);
                     assert_eq!($all(vec_c), false);
                 }
             }
         }
     }

    test_bool_red!(
        [i8x16_boolean_reductions, v128_any_true, i8x16_all_true]
            | [1_i8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            | [0_i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            | [1_i8, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    );
    test_bool_red!(
        [i16x8_boolean_reductions, v128_any_true, i16x8_all_true]
            | [1_i16, 1, 1, 1, 1, 1, 1, 1]
            | [0_i16, 0, 0, 0, 0, 0, 0, 0]
            | [1_i16, 0, 1, 0, 1, 0, 1, 0]
    );
    test_bool_red!(
        [i32x4_boolean_reductions, v128_any_true, i32x4_all_true]
            | [1_i32, 1, 1, 1]
            | [0_i32, 0, 0, 0]
            | [1_i32, 0, 1, 0]
    );
    test_bool_red!(
        [i64x2_boolean_reductions, v128_any_true, i64x2_all_true]
            | [1_i64, 1]
            | [0_i64, 0]
            | [1_i64, 0]
    );

    test_bop!(i8x16[i8; 16] | i8x16_eq[i8x16_eq_test]:
              ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
               [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]) =>
              [-1, 0, -1, 0 ,-1, 0, -1, -1, -1, 0, -1, 0 ,-1, 0, -1, -1]);
    test_bop!(i16x8[i16; 8] | i16x8_eq[i16x8_eq_test]:
               ([0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 2, 4, 4, 6, 6, 7]) =>
               [-1, 0, -1, 0 ,-1, 0, -1, -1]);
    test_bop!(i32x4[i32; 4] | i32x4_eq[i32x4_eq_test]:
               ([0, 1, 2, 3], [0, 2, 2, 4]) => [-1, 0, -1, 0]);
    test_bop!(i64x2[i64; 2] | i64x2_eq[i64x2_eq_test]:
               ([0, 1], [0, 2]) => [-1, 0]);
    test_bop!(f32x4[f32; 4] => i32 | f32x4_eq[f32x4_eq_test]:
               ([0., 1., 2., 3.], [0., 2., 2., 4.]) => [-1, 0, -1, 0]);
    test_bop!(f64x2[f64; 2] => i64 | f64x2_eq[f64x2_eq_test]: ([0., 1.], [0., 2.]) => [-1, 0]);

    test_bop!(i8x16[i8; 16] | i8x16_ne[i8x16_ne_test]:
               ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]) =>
               [0, -1, 0, -1 ,0, -1, 0, 0, 0, -1, 0, -1 ,0, -1, 0, 0]);
    test_bop!(i16x8[i16; 8] | i16x8_ne[i16x8_ne_test]:
               ([0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 2, 4, 4, 6, 6, 7]) =>
               [0, -1, 0, -1 ,0, -1, 0, 0]);
    test_bop!(i32x4[i32; 4] | i32x4_ne[i32x4_ne_test]:
               ([0, 1, 2, 3], [0, 2, 2, 4]) => [0, -1, 0, -1]);
    test_bop!(i64x2[i64; 2] | i64x2_ne[i64x2_ne_test]:
               ([0, 1], [0, 2]) => [0, -1]);
    test_bop!(f32x4[f32; 4] => i32 | f32x4_ne[f32x4_ne_test]:
               ([0., 1., 2., 3.], [0., 2., 2., 4.]) => [0, -1, 0, -1]);
    test_bop!(f64x2[f64; 2] => i64 | f64x2_ne[f64x2_ne_test]: ([0., 1.], [0., 2.]) => [0, -1]);

    test_bop!(i8x16[i8; 16] | i8x16_lt[i8x16_lt_s_test]:
               ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -12, 13, 14, 15],
                [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]) =>
               [0, -1, 0, -1 ,0, -1, 0, 0, 0, -1, 0, -1, -1, -1, 0, 0]);
    test_bop!(i8x16[i8; 16] | u8x16_lt[i8x16_lt_u_test]:
               ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -12, 13, 14, 15],
                [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]) =>
               [0, -1, 0, -1 ,0, -1, 0, 0, 0, -1, 0, -1 ,0, -1, 0, 0]);
    test_bop!(i16x8[i16; 8] | i16x8_lt[i16x8_lt_s_test]:
               ([0, 1, 2, 3, 4, 5, 6, -7], [0, 2, 2, 4, 4, 6, 6, 7]) =>
               [0, -1, 0, -1 ,0, -1, 0, -1]);
    test_bop!(i16x8[i16; 8] | u16x8_lt[i16x8_lt_u_test]:
               ([0, 1, 2, 3, 4, 5, 6, -7], [0, 2, 2, 4, 4, 6, 6, 7]) =>
               [0, -1, 0, -1 ,0, -1, 0, 0]);
    test_bop!(i32x4[i32; 4] | i32x4_lt[i32x4_lt_s_test]:
               ([-1, 1, 2, 3], [0, 2, 2, 4]) => [-1, -1, 0, -1]);
    test_bop!(i32x4[i32; 4] | u32x4_lt[i32x4_lt_u_test]:
               ([-1, 1, 2, 3], [0, 2, 2, 4]) => [0, -1, 0, -1]);
    test_bop!(i64x2[i64; 2] | i64x2_lt[i64x2_lt_s_test]:
               ([-1, 3], [0, 2]) => [-1, 0]);
    test_bop!(f32x4[f32; 4] => i32 | f32x4_lt[f32x4_lt_test]:
               ([0., 1., 2., 3.], [0., 2., 2., 4.]) => [0, -1, 0, -1]);
    test_bop!(f64x2[f64; 2] => i64 | f64x2_lt[f64x2_lt_test]: ([0., 1.], [0., 2.]) => [0, -1]);

    test_bop!(i8x16[i8; 16] | i8x16_gt[i8x16_gt_s_test]:
           ([0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, -15],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) =>
               [0, -1, 0, -1 ,0, -1, 0, 0, 0, -1, 0, -1 ,0, -1, 0, 0]);
    test_bop!(i8x16[i8; 16] | u8x16_gt[i8x16_gt_u_test]:
           ([0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, -15],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) =>
               [0, -1, 0, -1 ,0, -1, 0, 0, 0, -1, 0, -1 ,0, -1, 0, -1]);
    test_bop!(i16x8[i16; 8] | i16x8_gt[i16x8_gt_s_test]:
               ([0, 2, 2, 4, 4, 6, 6, -7], [0, 1, 2, 3, 4, 5, 6, 7]) =>
               [0, -1, 0, -1 ,0, -1, 0, 0]);
    test_bop!(i16x8[i16; 8] | u16x8_gt[i16x8_gt_u_test]:
               ([0, 2, 2, 4, 4, 6, 6, -7], [0, 1, 2, 3, 4, 5, 6, 7]) =>
               [0, -1, 0, -1 ,0, -1, 0, -1]);
    test_bop!(i32x4[i32; 4] | i32x4_gt[i32x4_gt_s_test]:
               ([0, 2, 2, -4], [0, 1, 2, 3]) => [0, -1, 0, 0]);
    test_bop!(i32x4[i32; 4] | u32x4_gt[i32x4_gt_u_test]:
               ([0, 2, 2, -4], [0, 1, 2, 3]) => [0, -1, 0, -1]);
    test_bop!(i64x2[i64; 2] | i64x2_gt[i64x2_gt_s_test]:
               ([-1, 2], [0, 1]) => [0, -1]);
    test_bop!(f32x4[f32; 4] => i32 | f32x4_gt[f32x4_gt_test]:
               ([0., 2., 2., 4.], [0., 1., 2., 3.]) => [0, -1, 0, -1]);
    test_bop!(f64x2[f64; 2] => i64 | f64x2_gt[f64x2_gt_test]: ([0., 2.], [0., 1.]) => [0, -1]);

    test_bop!(i8x16[i8; 16] | i8x16_ge[i8x16_ge_s_test]:
               ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, -15],
                [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]) =>
               [-1, 0, -1, 0 ,-1, 0, -1, -1, -1, 0, -1, 0 ,-1, 0, -1, 0]);
    test_bop!(i8x16[i8; 16] | u8x16_ge[i8x16_ge_u_test]:
               ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, -15],
                [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]) =>
               [-1, 0, -1, 0 ,-1, 0, -1, -1, -1, 0, -1, 0 ,-1, 0, -1, -1]);
    test_bop!(i16x8[i16; 8] | i16x8_ge[i16x8_ge_s_test]:
               ([0, 1, 2, 3, 4, 5, 6, -7], [0, 2, 2, 4, 4, 6, 6, 7]) =>
               [-1, 0, -1, 0 ,-1, 0, -1, 0]);
    test_bop!(i16x8[i16; 8] | u16x8_ge[i16x8_ge_u_test]:
               ([0, 1, 2, 3, 4, 5, 6, -7], [0, 2, 2, 4, 4, 6, 6, 7]) =>
               [-1, 0, -1, 0 ,-1, 0, -1, -1]);
    test_bop!(i32x4[i32; 4] | i32x4_ge[i32x4_ge_s_test]:
               ([0, 1, 2, -3], [0, 2, 2, 4]) => [-1, 0, -1, 0]);
    test_bop!(i32x4[i32; 4] | u32x4_ge[i32x4_ge_u_test]:
               ([0, 1, 2, -3], [0, 2, 2, 4]) => [-1, 0, -1, -1]);
    test_bop!(i64x2[i64; 2] | i64x2_ge[i64x2_ge_s_test]:
               ([0, 1], [-1, 2]) => [-1, 0]);
    test_bop!(f32x4[f32; 4] => i32 | f32x4_ge[f32x4_ge_test]:
               ([0., 1., 2., 3.], [0., 2., 2., 4.]) => [-1, 0, -1, 0]);
    test_bop!(f64x2[f64; 2] => i64 | f64x2_ge[f64x2_ge_test]: ([0., 1.], [0., 2.]) => [-1, 0]);

    test_bop!(i8x16[i8; 16] | i8x16_le[i8x16_le_s_test]:
               ([0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, -15],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
               ) =>
               [-1, 0, -1, 0 ,-1, 0, -1, -1, -1, 0, -1, 0 ,-1, 0, -1, -1]);
    test_bop!(i8x16[i8; 16] | u8x16_le[i8x16_le_u_test]:
               ([0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, -15],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
               ) =>
               [-1, 0, -1, 0 ,-1, 0, -1, -1, -1, 0, -1, 0 ,-1, 0, -1, 0]);
    test_bop!(i16x8[i16; 8] | i16x8_le[i16x8_le_s_test]:
               ([0, 2, 2, 4, 4, 6, 6, -7], [0, 1, 2, 3, 4, 5, 6, 7]) =>
               [-1, 0, -1, 0 ,-1, 0, -1, -1]);
    test_bop!(i16x8[i16; 8] | u16x8_le[i16x8_le_u_test]:
               ([0, 2, 2, 4, 4, 6, 6, -7], [0, 1, 2, 3, 4, 5, 6, 7]) =>
               [-1, 0, -1, 0 ,-1, 0, -1, 0]);
    test_bop!(i32x4[i32; 4] | i32x4_le[i32x4_le_s_test]:
               ([0, 2, 2, -4], [0, 1, 2, 3]) => [-1, 0, -1, -1]);
    test_bop!(i32x4[i32; 4] | u32x4_le[i32x4_le_u_test]:
               ([0, 2, 2, -4], [0, 1, 2, 3]) => [-1, 0, -1, 0]);
    test_bop!(i64x2[i64; 2] | i64x2_le[i64x2_le_s_test]:
               ([0, 2], [0, 1]) => [-1, 0]);
    test_bop!(f32x4[f32; 4] => i32 | f32x4_le[f32x4_le_test]:
               ([0., 2., 2., 4.], [0., 1., 2., 3.]) => [-1, 0, -1, -0]);
    test_bop!(f64x2[f64; 2] => i64 | f64x2_le[f64x2_le_test]: ([0., 2.], [0., 1.]) => [-1, 0]);

    test_uop!(f32x4[f32; 4] | f32x4_neg[f32x4_neg_test]: [0., 1., 2., 3.] => [ 0., -1., -2., -3.]);
    test_uop!(f32x4[f32; 4] | f32x4_abs[f32x4_abs_test]: [0., -1., 2., -3.] => [ 0., 1., 2., 3.]);
    test_bop!(f32x4[f32; 4] | f32x4_min[f32x4_min_test]:
              ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [0., -3., -4., 8.]);
    test_bop!(f32x4[f32; 4] | f32x4_min[f32x4_min_test_nan]:
              ([0., -1., 7., 8.], [1., -3., -4., f32::NAN])
              => [0., -3., -4., f32::NAN]);
    test_bop!(f32x4[f32; 4] | f32x4_max[f32x4_max_test]:
              ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [1., -1., 7., 10.]);
    test_bop!(f32x4[f32; 4] | f32x4_max[f32x4_max_test_nan]:
              ([0., -1., 7., 8.], [1., -3., -4., f32::NAN])
              => [1., -1., 7., f32::NAN]);
    test_bop!(f32x4[f32; 4] | f32x4_add[f32x4_add_test]:
              ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [1., -4., 3., 18.]);
    test_bop!(f32x4[f32; 4] | f32x4_sub[f32x4_sub_test]:
              ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [-1., 2., 11., -2.]);
    test_bop!(f32x4[f32; 4] | f32x4_mul[f32x4_mul_test]:
              ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [0., 3., -28., 80.]);
    test_bop!(f32x4[f32; 4] | f32x4_div[f32x4_div_test]:
              ([0., -8., 70., 8.], [1., 4., 10., 2.]) => [0., -2., 7., 4.]);

    test_uop!(f64x2[f64; 2] | f64x2_neg[f64x2_neg_test]: [0., 1.] => [ 0., -1.]);
    test_uop!(f64x2[f64; 2] | f64x2_abs[f64x2_abs_test]: [0., -1.] => [ 0., 1.]);
    test_bop!(f64x2[f64; 2] | f64x2_min[f64x2_min_test]:
               ([0., -1.], [1., -3.]) => [0., -3.]);
    test_bop!(f64x2[f64; 2] | f64x2_min[f64x2_min_test_nan]:
               ([7., 8.], [-4., f64::NAN])
               => [ -4., f64::NAN]);
    test_bop!(f64x2[f64; 2] | f64x2_max[f64x2_max_test]:
               ([0., -1.], [1., -3.]) => [1., -1.]);
    test_bop!(f64x2[f64; 2] | f64x2_max[f64x2_max_test_nan]:
               ([7., 8.], [ -4., f64::NAN])
               => [7., f64::NAN]);
    test_bop!(f64x2[f64; 2] | f64x2_add[f64x2_add_test]:
               ([0., -1.], [1., -3.]) => [1., -4.]);
    test_bop!(f64x2[f64; 2] | f64x2_sub[f64x2_sub_test]:
               ([0., -1.], [1., -3.]) => [-1., 2.]);
    test_bop!(f64x2[f64; 2] | f64x2_mul[f64x2_mul_test]:
               ([0., -1.], [1., -3.]) => [0., 3.]);
    test_bop!(f64x2[f64; 2] | f64x2_div[f64x2_div_test]:
               ([0., -8.], [1., 4.]) => [0., -2.]);

    macro_rules! test_conv {
        ($test_id:ident | $conv_id:ident | $to_ty:ident | $from:expr,  $to:expr) => {
            #[test]
            fn $test_id() {
                unsafe {
                    let from: v128 = transmute($from);
                    let to: v128 = transmute($to);

                    let r: v128 = $conv_id(from);

                    compare_bytes(r, to);
                }
            }
        };
    }

    test_conv!(
        f32x4_convert_s_i32x4 | f32x4_convert_i32x4 | f32x4 | [1_i32, 2, 3, 4],
        [1_f32, 2., 3., 4.]
    );
    test_conv!(
        f32x4_convert_u_i32x4 | f32x4_convert_u32x4 | f32x4 | [u32::MAX, 2, 3, 4],
        [u32::MAX as f32, 2., 3., 4.]
    );

    #[test]
    fn test_conversions() {
        compare_bytes(
            i32x4_trunc_sat_f32x4(f32x4(1., f32::NEG_INFINITY, f32::INFINITY, f32::NAN)),
            i32x4(1, i32::MIN, i32::MAX, 0),
        );
        compare_bytes(
            u32x4_trunc_sat_f32x4(f32x4(1., f32::NEG_INFINITY, f32::INFINITY, f32::NAN)),
            u32x4(1, 0, u32::MAX, 0),
        );
        compare_bytes(f64x2_convert_low_i32x4(i32x4(1, 2, 3, 4)), f64x2(1., 2.));
        compare_bytes(
            f64x2_convert_low_i32x4(i32x4(i32::MIN, i32::MAX, 3, 4)),
            f64x2(f64::from(i32::MIN), f64::from(i32::MAX)),
        );
        compare_bytes(f64x2_convert_low_u32x4(u32x4(1, 2, 3, 4)), f64x2(1., 2.));
        compare_bytes(
            f64x2_convert_low_u32x4(u32x4(u32::MIN, u32::MAX, 3, 4)),
            f64x2(f64::from(u32::MIN), f64::from(u32::MAX)),
        );

        compare_bytes(
            i32x4_trunc_sat_f64x2_zero(f64x2(1., f64::NEG_INFINITY)),
            i32x4(1, i32::MIN, 0, 0),
        );
        compare_bytes(
            i32x4_trunc_sat_f64x2_zero(f64x2(f64::NAN, f64::INFINITY)),
            i32x4(0, i32::MAX, 0, 0),
        );
        compare_bytes(
            u32x4_trunc_sat_f64x2_zero(f64x2(1., f64::NEG_INFINITY)),
            u32x4(1, 0, 0, 0),
        );
        compare_bytes(
            u32x4_trunc_sat_f64x2_zero(f64x2(f64::NAN, f64::INFINITY)),
            u32x4(0, u32::MAX, 0, 0),
        );
    }

    #[test]
    fn test_popcnt() {
        unsafe {
            for i in 0..=255 {
                compare_bytes(
                    i8x16_popcnt(u8x16_splat(i)),
                    u8x16_splat(i.count_ones() as u8),
                )
            }

            let vectors = [
                [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                [
                    100, 200, 50, 0, 10, 7, 38, 185, 192, 3, 34, 85, 93, 7, 31, 99,
                ],
            ];

            for vector in vectors.iter() {
                compare_bytes(
                    i8x16_popcnt(transmute(*vector)),
                    i8x16(
                        vector[0].count_ones() as i8,
                        vector[1].count_ones() as i8,
                        vector[2].count_ones() as i8,
                        vector[3].count_ones() as i8,
                        vector[4].count_ones() as i8,
                        vector[5].count_ones() as i8,
                        vector[6].count_ones() as i8,
                        vector[7].count_ones() as i8,
                        vector[8].count_ones() as i8,
                        vector[9].count_ones() as i8,
                        vector[10].count_ones() as i8,
                        vector[11].count_ones() as i8,
                        vector[12].count_ones() as i8,
                        vector[13].count_ones() as i8,
                        vector[14].count_ones() as i8,
                        vector[15].count_ones() as i8,
                    ),
                )
            }
        }
    }

    #[test]
    fn test_promote_demote() {
        let tests = [
            [1., 2.],
            [f64::NAN, f64::INFINITY],
            [100., 201.],
            [0., -0.],
            [f64::NEG_INFINITY, 0.],
        ];

        for [a, b] in tests {
            compare_bytes(
                f32x4_demote_f64x2_zero(f64x2(a, b)),
                f32x4(a as f32, b as f32, 0., 0.),
            );
            compare_bytes(
                f64x2_promote_low_f32x4(f32x4(a as f32, b as f32, 0., 0.)),
                f64x2(a, b),
            );
        }
    }

    #[test]
    fn test_extmul() {
        macro_rules! test {
            ($(
                $ctor:ident {
                    from: $from:ident,
                    to: $to:ident,
                    low: $low:ident,
                    high: $high:ident,
                } => {
                    $(([$($a:tt)*] * [$($b:tt)*]))*
                }
            )*) => ($(
                $(unsafe {
                    let a: [$from; 16 / mem::size_of::<$from>()] = [$($a)*];
                    let b: [$from; 16 / mem::size_of::<$from>()] = [$($b)*];
                    let low = mem::transmute::<_, [$to; 16 / mem::size_of::<$to>()]>($low($ctor($($a)*), $ctor($($b)*)));
                    let high = mem::transmute::<_, [$to; 16 / mem::size_of::<$to>()]>($high($ctor($($a)*), $ctor($($b)*)));

                    let half = a.len() / 2;
                    for i in 0..half {
                        assert_eq!(
                            (a[i] as $to).wrapping_mul((b[i] as $to)),
                            low[i],
                            "expected {} * {}", a[i] as $to, b[i] as $to,
                        );
                        assert_eq!(
                            (a[half + i] as $to).wrapping_mul((b[half + i] as $to)),
                            high[i],
                            "expected {} * {}", a[half + i] as $to, b[half + i] as $to,
                        );
                    }
                })*
            )*)
        }
        test! {
            i8x16 {
                from: i8,
                to: i16,
                low: i16x8_extmul_low_i8x16,
                high: i16x8_extmul_high_i8x16,
            } => {
                (
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        *
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                )
                (
                    [-1, -2, 3, 100, 124, -38, 33, 87, 92, 108, 22, 8, -43, -128, 22, 0]
                        *
                    [-5, -2, 6, 10, 45, -4, 4, -2, 0, 88, 92, -102, -98, 83, 73, 54]
                )
            }
            u8x16 {
                from: u8,
                to: u16,
                low: u16x8_extmul_low_u8x16,
                high: u16x8_extmul_high_u8x16,
            } => {
                (
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        *
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                )
                (
                    [1, 2, 3, 100, 124, 38, 33, 87, 92, 198, 22, 8, 43, 128, 22, 0]
                        *
                    [5, 200, 6, 10, 45, 248, 4, 2, 0, 2, 92, 102, 234, 83, 73, 54]
                )
            }
            i16x8 {
                from: i16,
                to: i32,
                low: i32x4_extmul_low_i16x8,
                high: i32x4_extmul_high_i16x8,
            } => {
                (
                    [0, 0, 0, 0, 0, 0, 0, 0]
                        *
                    [0, 0, 0, 0, 0, 0, 0, 0]
                )
                (
                    [-1, 0, i16::MAX, 19931, -2259, 64, 200, 87]
                        *
                    [1, 1, i16::MIN, 29391, 105, 2, 100, -2]
                )
            }
            u16x8 {
                from: u16,
                to: u32,
                low: u32x4_extmul_low_u16x8,
                high: u32x4_extmul_high_u16x8,
            } => {
                (
                    [0, 0, 0, 0, 0, 0, 0, 0]
                        *
                    [0, 0, 0, 0, 0, 0, 0, 0]
                )
                (
                    [1, 0, u16::MAX, 19931, 2259, 64, 200, 87]
                        *
                    [1, 1, 3, 29391, 105, 2, 100, 2]
                )
            }
            i32x4 {
                from: i32,
                to: i64,
                low: i64x2_extmul_low_i32x4,
                high: i64x2_extmul_high_i32x4,
            } => {
                (
                    [0, 0, 0, 0]
                        *
                    [0, 0, 0, 0]
                )
                (
                    [-1, 0, i32::MAX, 19931]
                        *
                    [1, 1, i32::MIN, 29391]
                )
                (
                    [i32::MAX, 3003183, 3 << 20, 0xffffff]
                        *
                    [i32::MAX, i32::MIN, -40042, 300]
                )
            }
            u32x4 {
                from: u32,
                to: u64,
                low: u64x2_extmul_low_u32x4,
                high: u64x2_extmul_high_u32x4,
            } => {
                (
                    [0, 0, 0, 0]
                        *
                    [0, 0, 0, 0]
                )
                (
                    [1, 0, u32::MAX, 19931]
                        *
                    [1, 1, 3, 29391]
                )
                (
                    [u32::MAX, 3003183, 3 << 20, 0xffffff]
                        *
                    [u32::MAX, 3000, 40042, 300]
                )
            }
        }
    }

    #[test]
    fn test_q15mulr_sat_s() {
        fn test(a: [i16; 8], b: [i16; 8]) {
            let a_v = i16x8(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
            let b_v = i16x8(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);
            let result = i16x8_q15mulr_sat(a_v, b_v);
            let result = unsafe { mem::transmute::<v128, [i16; 8]>(result) };

            for (i, (a, b)) in a.iter().zip(&b).enumerate() {
                assert_eq!(
                    result[i],
                    (((*a as i32) * (*b as i32) + 0x4000) >> 15) as i16
                );
            }
        }

        test([0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]);
        test([1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]);
        test(
            [-1, 100, 2003, -29494, 12, 128, 994, 1],
            [-4049, 8494, -10483, 0, 5, 2222, 883, -9],
        );
    }

    #[test]
    fn test_extadd() {
        macro_rules! test {
            ($(
                $func:ident {
                    from: $from:ident,
                    to: $to:ident,
                } => {
                    $([$($a:tt)*])*
                }
            )*) => ($(
                $(unsafe {
                    let a: [$from; 16 / mem::size_of::<$from>()] = [$($a)*];
                    let a_v = mem::transmute::<_, v128>(a);
                    let r = mem::transmute::<v128, [$to; 16 / mem::size_of::<$to>()]>($func(a_v));

                    let half = a.len() / 2;
                    for i in 0..half {
                        assert_eq!(
                            (a[2 * i] as $to).wrapping_add((a[2 * i + 1] as $to)),
                            r[i],
                            "failed {} + {} != {}",
                            a[2 * i] as $to,
                            a[2 * i + 1] as $to,
                            r[i],
                        );
                    }
                })*
            )*)
        }
        test! {
            i16x8_extadd_pairwise_i8x16 {
                from: i8,
                to: i16,
            } => {
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                [-1, -2, 3, 100, 124, -38, 33, 87, 92, 108, 22, 8, -43, -128, 22, 0]
                [-5, -2, 6, 10, 45, -4, 4, -2, 0, 88, 92, -102, -98, 83, 73, 54]
            }
            i16x8_extadd_pairwise_u8x16 {
                from: u8,
                to: i16,
            } => {
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                [1, 2, 3, 100, 124, 38, 33, 87, 92, 198, 22, 8, 43, 128, 22, 0]
                [5, 200, 6, 10, 45, 248, 4, 2, 0, 2, 92, 102, 234, 83, 73, 54]
            }
            i32x4_extadd_pairwise_i16x8 {
                from: i16,
                to: i32,
            } => {
                [0, 0, 0, 0, 0, 0, 0, 0]
                [-1, 0, i16::MAX, 19931, -2259, 64, 200, 87]
                [1, 1, i16::MIN, 29391, 105, 2, 100, -2]
            }
            i32x4_extadd_pairwise_u16x8 {
                from: u16,
                to: i32,
            } => {
                [0, 0, 0, 0, 0, 0, 0, 0]
                [1, 0, u16::MAX, 19931, 2259, 64, 200, 87]
                [1, 1, 3, 29391, 105, 2, 100, 2]
            }
        }
    }
}
