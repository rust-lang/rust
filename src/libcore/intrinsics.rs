// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! rustc compiler intrinsics.
//!
//! The corresponding definitions are in librustc_trans/trans/intrinsic.rs.
//!
//! # Volatiles
//!
//! The volatile intrinsics provide operations intended to act on I/O
//! memory, which are guaranteed to not be reordered by the compiler
//! across other volatile intrinsics. See the LLVM documentation on
//! [[volatile]].
//!
//! [volatile]: http://llvm.org/docs/LangRef.html#volatile-memory-accesses
//!
//! # Atomics
//!
//! The atomic intrinsics provide common atomic operations on machine
//! words, with multiple possible memory orderings. They obey the same
//! semantics as C++11. See the LLVM documentation on [[atomics]].
//!
//! [atomics]: http://llvm.org/docs/Atomics.html
//!
//! A quick refresher on memory ordering:
//!
//! * Acquire - a barrier for acquiring a lock. Subsequent reads and writes
//!   take place after the barrier.
//! * Release - a barrier for releasing a lock. Preceding reads and writes
//!   take place before the barrier.
//! * Sequentially consistent - sequentially consistent operations are
//!   guaranteed to happen in order. This is the standard mode for working
//!   with atomic types and is equivalent to Java's `volatile`.

#![unstable(feature = "core")]
#![allow(missing_docs)]

use marker::Sized;

pub type GlueFn = extern "Rust" fn(*const i8);

#[lang="ty_desc"]
#[derive(Copy)]
pub struct TyDesc {
    // sizeof(T)
    pub size: usize,

    // alignof(T)
    pub align: usize,

    // Called when a value of type `T` is no longer needed
    pub drop_glue: GlueFn,

    // Name corresponding to the type
    pub name: &'static str,
}

extern "rust-intrinsic" {

    // NB: These intrinsics take unsafe pointers because they mutate aliased
    // memory, which is not valid for either `&` or `&mut`.

    pub fn atomic_cxchg<T>(dst: *mut T, old: T, src: T) -> T;
    pub fn atomic_cxchg_acq<T>(dst: *mut T, old: T, src: T) -> T;
    pub fn atomic_cxchg_rel<T>(dst: *mut T, old: T, src: T) -> T;
    pub fn atomic_cxchg_acqrel<T>(dst: *mut T, old: T, src: T) -> T;
    pub fn atomic_cxchg_relaxed<T>(dst: *mut T, old: T, src: T) -> T;

    pub fn atomic_load<T>(src: *const T) -> T;
    pub fn atomic_load_acq<T>(src: *const T) -> T;
    pub fn atomic_load_relaxed<T>(src: *const T) -> T;
    pub fn atomic_load_unordered<T>(src: *const T) -> T;

    pub fn atomic_store<T>(dst: *mut T, val: T);
    pub fn atomic_store_rel<T>(dst: *mut T, val: T);
    pub fn atomic_store_relaxed<T>(dst: *mut T, val: T);
    pub fn atomic_store_unordered<T>(dst: *mut T, val: T);

    pub fn atomic_xchg<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_xchg_acq<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_xchg_rel<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_xchg_acqrel<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_xchg_relaxed<T>(dst: *mut T, src: T) -> T;

    pub fn atomic_xadd<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_xadd_acq<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_xadd_rel<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_xadd_acqrel<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_xadd_relaxed<T>(dst: *mut T, src: T) -> T;

    pub fn atomic_xsub<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_xsub_acq<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_xsub_rel<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_xsub_acqrel<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_xsub_relaxed<T>(dst: *mut T, src: T) -> T;

    pub fn atomic_and<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_and_acq<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_and_rel<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_and_acqrel<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_and_relaxed<T>(dst: *mut T, src: T) -> T;

    pub fn atomic_nand<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_nand_acq<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_nand_rel<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_nand_acqrel<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_nand_relaxed<T>(dst: *mut T, src: T) -> T;

    pub fn atomic_or<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_or_acq<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_or_rel<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_or_acqrel<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_or_relaxed<T>(dst: *mut T, src: T) -> T;

    pub fn atomic_xor<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_xor_acq<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_xor_rel<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_xor_acqrel<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_xor_relaxed<T>(dst: *mut T, src: T) -> T;

    pub fn atomic_max<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_max_acq<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_max_rel<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_max_acqrel<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_max_relaxed<T>(dst: *mut T, src: T) -> T;

    pub fn atomic_min<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_min_acq<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_min_rel<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_min_acqrel<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_min_relaxed<T>(dst: *mut T, src: T) -> T;

    pub fn atomic_umin<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_umin_acq<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_umin_rel<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_umin_acqrel<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_umin_relaxed<T>(dst: *mut T, src: T) -> T;

    pub fn atomic_umax<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_umax_acq<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_umax_rel<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_umax_acqrel<T>(dst: *mut T, src: T) -> T;
    pub fn atomic_umax_relaxed<T>(dst: *mut T, src: T) -> T;
}

extern "rust-intrinsic" {

    pub fn atomic_fence();
    pub fn atomic_fence_acq();
    pub fn atomic_fence_rel();
    pub fn atomic_fence_acqrel();

    /// Abort the execution of the process.
    pub fn abort() -> !;

    /// Tell LLVM that this point in the code is not reachable,
    /// enabling further optimizations.
    ///
    /// NB: This is very different from the `unreachable!()` macro!
    pub fn unreachable() -> !;

    /// Inform the optimizer that a condition is always true.
    /// If the condition is false, the behavior is undefined.
    ///
    /// No code is generated for this intrinsic, but the optimizer will try
    /// to preserve it (and its condition) between passes, which may interfere
    /// with optimization of surrounding code and reduce performance. It should
    /// not be used if the invariant can be discovered by the optimizer on its
    /// own, or if it does not enable any significant optimizations.
    pub fn assume(b: bool);

    /// Execute a breakpoint trap, for inspection by a debugger.
    pub fn breakpoint();

    /// The size of a type in bytes.
    ///
    /// This is the exact number of bytes in memory taken up by a
    /// value of the given type. In other words, a memset of this size
    /// would *exactly* overwrite a value. When laid out in vectors
    /// and structures there may be additional padding between
    /// elements.
    pub fn size_of<T>() -> usize;

    /// Move a value to an uninitialized memory location.
    ///
    /// Drop glue is not run on the destination.
    pub fn move_val_init<T>(dst: &mut T, src: T);

    pub fn min_align_of<T>() -> usize;
    pub fn pref_align_of<T>() -> usize;

    /// Get a static pointer to a type descriptor.
    pub fn get_tydesc<T: ?Sized>() -> *const TyDesc;

    /// Gets an identifier which is globally unique to the specified type. This
    /// function will return the same value for a type regardless of whichever
    /// crate it is invoked in.
    pub fn type_id<T: ?Sized + 'static>() -> u64;

    /// Create a value initialized to zero.
    ///
    /// `init` is unsafe because it returns a zeroed-out datum,
    /// which is unsafe unless T is Copy.
    pub fn init<T>() -> T;

    /// Create an uninitialized value.
    pub fn uninit<T>() -> T;

    /// Move a value out of scope without running drop glue.
    ///
    /// `forget` is unsafe because the caller is responsible for
    /// ensuring the argument is deallocated already.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn forget<T>(_: T) -> ();

    /// Unsafely transforms a value of one type into a value of another type.
    ///
    /// Both types must have the same size.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::mem;
    ///
    /// let v: &[u8] = unsafe { mem::transmute("L") };
    /// assert!(v == [76u8]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn transmute<T,U>(e: T) -> U;

    /// Gives the address for the return value of the enclosing function.
    ///
    /// Using this intrinsic in a function that does not use an out pointer
    /// will trigger a compiler error.
    pub fn return_address() -> *const u8;

    /// Returns `true` if a type requires drop glue.
    pub fn needs_drop<T>() -> bool;

    /// Returns `true` if a type is managed (will be allocated on the local heap)
    pub fn owns_managed<T>() -> bool;

    /// Calculates the offset from a pointer. The offset *must* be in-bounds of
    /// the object, or one-byte-past-the-end. An arithmetic overflow is also
    /// undefined behaviour.
    ///
    /// This is implemented as an intrinsic to avoid converting to and from an
    /// integer, since the conversion would throw away aliasing information.
    pub fn offset<T>(dst: *const T, offset: isize) -> *const T;

    /// Copies `count * size_of<T>` bytes from `src` to `dst`. The source
    /// and destination may *not* overlap.
    ///
    /// `copy_nonoverlapping_memory` is semantically equivalent to C's `memcpy`.
    ///
    /// # Safety
    ///
    /// Beyond requiring that the program must be allowed to access both regions
    /// of memory, it is Undefined Behaviour for source and destination to
    /// overlap. Care must also be taken with the ownership of `src` and
    /// `dst`. This method semantically moves the values of `src` into `dst`.
    /// However it does not drop the contents of `dst`, or prevent the contents
    /// of `src` from being dropped or used.
    ///
    /// # Examples
    ///
    /// A safe swap function:
    ///
    /// ```
    /// use std::mem;
    /// use std::ptr;
    ///
    /// fn swap<T>(x: &mut T, y: &mut T) {
    ///     unsafe {
    ///         // Give ourselves some scratch space to work with
    ///         let mut t: T = mem::uninitialized();
    ///
    ///         // Perform the swap, `&mut` pointers never alias
    ///         ptr::copy_nonoverlapping_memory(&mut t, &*x, 1);
    ///         ptr::copy_nonoverlapping_memory(x, &*y, 1);
    ///         ptr::copy_nonoverlapping_memory(y, &t, 1);
    ///
    ///         // y and t now point to the same thing, but we need to completely forget `tmp`
    ///         // because it's no longer relevant.
    ///         mem::forget(t);
    ///     }
    /// }
    /// ```
    #[unstable(feature = "core")]
    pub fn copy_nonoverlapping_memory<T>(dst: *mut T, src: *const T, count: usize);

    /// Copies `count * size_of<T>` bytes from `src` to `dst`. The source
    /// and destination may overlap.
    ///
    /// `copy_memory` is semantically equivalent to C's `memmove`.
    ///
    /// # Safety
    ///
    /// Care must be taken with the ownership of `src` and `dst`.
    /// This method semantically moves the values of `src` into `dst`.
    /// However it does not drop the contents of `dst`, or prevent the contents of `src`
    /// from being dropped or used.
    ///
    /// # Examples
    ///
    /// Efficiently create a Rust vector from an unsafe buffer:
    ///
    /// ```
    /// use std::ptr;
    ///
    /// unsafe fn from_buf_raw<T>(ptr: *const T, elts: uint) -> Vec<T> {
    ///     let mut dst = Vec::with_capacity(elts);
    ///     dst.set_len(elts);
    ///     ptr::copy_memory(dst.as_mut_ptr(), ptr, elts);
    ///     dst
    /// }
    /// ```
    ///
    #[unstable(feature = "core")]
    pub fn copy_memory<T>(dst: *mut T, src: *const T, count: usize);

    /// Invokes memset on the specified pointer, setting `count * size_of::<T>()`
    /// bytes of memory starting at `dst` to `c`.
    #[unstable(feature = "core",
               reason = "uncertain about naming and semantics")]
    pub fn set_memory<T>(dst: *mut T, val: u8, count: usize);

    /// Equivalent to the appropriate `llvm.memcpy.p0i8.0i8.*` intrinsic, with
    /// a size of `count` * `size_of::<T>()` and an alignment of
    /// `min_align_of::<T>()`
    ///
    /// The volatile parameter parameter is set to `true`, so it will not be optimized out.
    pub fn volatile_copy_nonoverlapping_memory<T>(dst: *mut T, src: *const T,
                                                  count: usize);
    /// Equivalent to the appropriate `llvm.memmove.p0i8.0i8.*` intrinsic, with
    /// a size of `count` * `size_of::<T>()` and an alignment of
    /// `min_align_of::<T>()`
    ///
    /// The volatile parameter parameter is set to `true`, so it will not be optimized out.
    pub fn volatile_copy_memory<T>(dst: *mut T, src: *const T, count: usize);
    /// Equivalent to the appropriate `llvm.memset.p0i8.*` intrinsic, with a
    /// size of `count` * `size_of::<T>()` and an alignment of
    /// `min_align_of::<T>()`.
    ///
    /// The volatile parameter parameter is set to `true`, so it will not be optimized out.
    pub fn volatile_set_memory<T>(dst: *mut T, val: u8, count: usize);

    /// Perform a volatile load from the `src` pointer.
    pub fn volatile_load<T>(src: *const T) -> T;
    /// Perform a volatile store to the `dst` pointer.
    pub fn volatile_store<T>(dst: *mut T, val: T);

    /// Returns the square root of an `f32`
    pub fn sqrtf32(x: f32) -> f32;
    /// Returns the square root of an `f64`
    pub fn sqrtf64(x: f64) -> f64;

    /// Raises an `f32` to an integer power.
    pub fn powif32(a: f32, x: i32) -> f32;
    /// Raises an `f64` to an integer power.
    pub fn powif64(a: f64, x: i32) -> f64;

    /// Returns the sine of an `f32`.
    pub fn sinf32(x: f32) -> f32;
    /// Returns the sine of an `f64`.
    pub fn sinf64(x: f64) -> f64;

    /// Returns the cosine of an `f32`.
    pub fn cosf32(x: f32) -> f32;
    /// Returns the cosine of an `f64`.
    pub fn cosf64(x: f64) -> f64;

    /// Raises an `f32` to an `f32` power.
    pub fn powf32(a: f32, x: f32) -> f32;
    /// Raises an `f64` to an `f64` power.
    pub fn powf64(a: f64, x: f64) -> f64;

    /// Returns the exponential of an `f32`.
    pub fn expf32(x: f32) -> f32;
    /// Returns the exponential of an `f64`.
    pub fn expf64(x: f64) -> f64;

    /// Returns 2 raised to the power of an `f32`.
    pub fn exp2f32(x: f32) -> f32;
    /// Returns 2 raised to the power of an `f64`.
    pub fn exp2f64(x: f64) -> f64;

    /// Returns the natural logarithm of an `f32`.
    pub fn logf32(x: f32) -> f32;
    /// Returns the natural logarithm of an `f64`.
    pub fn logf64(x: f64) -> f64;

    /// Returns the base 10 logarithm of an `f32`.
    pub fn log10f32(x: f32) -> f32;
    /// Returns the base 10 logarithm of an `f64`.
    pub fn log10f64(x: f64) -> f64;

    /// Returns the base 2 logarithm of an `f32`.
    pub fn log2f32(x: f32) -> f32;
    /// Returns the base 2 logarithm of an `f64`.
    pub fn log2f64(x: f64) -> f64;

    /// Returns `a * b + c` for `f32` values.
    pub fn fmaf32(a: f32, b: f32, c: f32) -> f32;
    /// Returns `a * b + c` for `f64` values.
    pub fn fmaf64(a: f64, b: f64, c: f64) -> f64;

    /// Returns the absolute value of an `f32`.
    pub fn fabsf32(x: f32) -> f32;
    /// Returns the absolute value of an `f64`.
    pub fn fabsf64(x: f64) -> f64;

    /// Copies the sign from `y` to `x` for `f32` values.
    pub fn copysignf32(x: f32, y: f32) -> f32;
    /// Copies the sign from `y` to `x` for `f64` values.
    pub fn copysignf64(x: f64, y: f64) -> f64;

    /// Returns the largest integer less than or equal to an `f32`.
    pub fn floorf32(x: f32) -> f32;
    /// Returns the largest integer less than or equal to an `f64`.
    pub fn floorf64(x: f64) -> f64;

    /// Returns the smallest integer greater than or equal to an `f32`.
    pub fn ceilf32(x: f32) -> f32;
    /// Returns the smallest integer greater than or equal to an `f64`.
    pub fn ceilf64(x: f64) -> f64;

    /// Returns the integer part of an `f32`.
    pub fn truncf32(x: f32) -> f32;
    /// Returns the integer part of an `f64`.
    pub fn truncf64(x: f64) -> f64;

    /// Returns the nearest integer to an `f32`. May raise an inexact floating-point exception
    /// if the argument is not an integer.
    pub fn rintf32(x: f32) -> f32;
    /// Returns the nearest integer to an `f64`. May raise an inexact floating-point exception
    /// if the argument is not an integer.
    pub fn rintf64(x: f64) -> f64;

    /// Returns the nearest integer to an `f32`.
    pub fn nearbyintf32(x: f32) -> f32;
    /// Returns the nearest integer to an `f64`.
    pub fn nearbyintf64(x: f64) -> f64;

    /// Returns the nearest integer to an `f32`. Rounds half-way cases away from zero.
    pub fn roundf32(x: f32) -> f32;
    /// Returns the nearest integer to an `f64`. Rounds half-way cases away from zero.
    pub fn roundf64(x: f64) -> f64;

    /// Returns the number of bits set in a `u8`.
    pub fn ctpop8(x: u8) -> u8;
    /// Returns the number of bits set in a `u16`.
    pub fn ctpop16(x: u16) -> u16;
    /// Returns the number of bits set in a `u32`.
    pub fn ctpop32(x: u32) -> u32;
    /// Returns the number of bits set in a `u64`.
    pub fn ctpop64(x: u64) -> u64;

    /// Returns the number of leading bits unset in a `u8`.
    pub fn ctlz8(x: u8) -> u8;
    /// Returns the number of leading bits unset in a `u16`.
    pub fn ctlz16(x: u16) -> u16;
    /// Returns the number of leading bits unset in a `u32`.
    pub fn ctlz32(x: u32) -> u32;
    /// Returns the number of leading bits unset in a `u64`.
    pub fn ctlz64(x: u64) -> u64;

    /// Returns the number of trailing bits unset in a `u8`.
    pub fn cttz8(x: u8) -> u8;
    /// Returns the number of trailing bits unset in a `u16`.
    pub fn cttz16(x: u16) -> u16;
    /// Returns the number of trailing bits unset in a `u32`.
    pub fn cttz32(x: u32) -> u32;
    /// Returns the number of trailing bits unset in a `u64`.
    pub fn cttz64(x: u64) -> u64;

    /// Reverses the bytes in a `u16`.
    pub fn bswap16(x: u16) -> u16;
    /// Reverses the bytes in a `u32`.
    pub fn bswap32(x: u32) -> u32;
    /// Reverses the bytes in a `u64`.
    pub fn bswap64(x: u64) -> u64;

    /// Performs checked `i8` addition.
    pub fn i8_add_with_overflow(x: i8, y: i8) -> (i8, bool);
    /// Performs checked `i16` addition.
    pub fn i16_add_with_overflow(x: i16, y: i16) -> (i16, bool);
    /// Performs checked `i32` addition.
    pub fn i32_add_with_overflow(x: i32, y: i32) -> (i32, bool);
    /// Performs checked `i64` addition.
    pub fn i64_add_with_overflow(x: i64, y: i64) -> (i64, bool);

    /// Performs checked `u8` addition.
    pub fn u8_add_with_overflow(x: u8, y: u8) -> (u8, bool);
    /// Performs checked `u16` addition.
    pub fn u16_add_with_overflow(x: u16, y: u16) -> (u16, bool);
    /// Performs checked `u32` addition.
    pub fn u32_add_with_overflow(x: u32, y: u32) -> (u32, bool);
    /// Performs checked `u64` addition.
    pub fn u64_add_with_overflow(x: u64, y: u64) -> (u64, bool);

    /// Performs checked `i8` subtraction.
    pub fn i8_sub_with_overflow(x: i8, y: i8) -> (i8, bool);
    /// Performs checked `i16` subtraction.
    pub fn i16_sub_with_overflow(x: i16, y: i16) -> (i16, bool);
    /// Performs checked `i32` subtraction.
    pub fn i32_sub_with_overflow(x: i32, y: i32) -> (i32, bool);
    /// Performs checked `i64` subtraction.
    pub fn i64_sub_with_overflow(x: i64, y: i64) -> (i64, bool);

    /// Performs checked `u8` subtraction.
    pub fn u8_sub_with_overflow(x: u8, y: u8) -> (u8, bool);
    /// Performs checked `u16` subtraction.
    pub fn u16_sub_with_overflow(x: u16, y: u16) -> (u16, bool);
    /// Performs checked `u32` subtraction.
    pub fn u32_sub_with_overflow(x: u32, y: u32) -> (u32, bool);
    /// Performs checked `u64` subtraction.
    pub fn u64_sub_with_overflow(x: u64, y: u64) -> (u64, bool);

    /// Performs checked `i8` multiplication.
    pub fn i8_mul_with_overflow(x: i8, y: i8) -> (i8, bool);
    /// Performs checked `i16` multiplication.
    pub fn i16_mul_with_overflow(x: i16, y: i16) -> (i16, bool);
    /// Performs checked `i32` multiplication.
    pub fn i32_mul_with_overflow(x: i32, y: i32) -> (i32, bool);
    /// Performs checked `i64` multiplication.
    pub fn i64_mul_with_overflow(x: i64, y: i64) -> (i64, bool);

    /// Performs checked `u8` multiplication.
    pub fn u8_mul_with_overflow(x: u8, y: u8) -> (u8, bool);
    /// Performs checked `u16` multiplication.
    pub fn u16_mul_with_overflow(x: u16, y: u16) -> (u16, bool);
    /// Performs checked `u32` multiplication.
    pub fn u32_mul_with_overflow(x: u32, y: u32) -> (u32, bool);
    /// Performs checked `u64` multiplication.
    pub fn u64_mul_with_overflow(x: u64, y: u64) -> (u64, bool);
}
