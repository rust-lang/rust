// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! rustc compiler intrinsics.

The corresponding definitions are in librustc/middle/trans/foreign.rs.

# Atomics

The atomic intrinsics provide common atomic operations on machine
words, with multiple possible memory orderings. They obey the same
semantics as C++0x. See the LLVM documentation on [[atomics]].

[atomics]: http://llvm.org/docs/Atomics.html

A quick refresher on memory ordering:

* Acquire - a barrier for aquiring a lock. Subsequent reads and writes
  take place after the barrier.
* Release - a barrier for releasing a lock. Preceding reads and writes
  take place before the barrier.
* Sequentially consistent - sequentially consistent operations are
  guaranteed to happen in order. This is the standard mode for working
  with atomic types and is equivalent to Java's `volatile`.

*/

#[abi = "rust-intrinsic"]
pub extern "rust-intrinsic" {

    /// Atomic compare and exchange, sequentially consistent.
    pub fn atomic_cxchg(dst: &mut int, old: int, src: int) -> int;
    /// Atomic compare and exchange, acquire ordering.
    pub fn atomic_cxchg_acq(dst: &mut int, old: int, src: int) -> int;
    /// Atomic compare and exchange, release ordering.
    pub fn atomic_cxchg_rel(dst: &mut int, old: int, src: int) -> int;

    /// Atomic load, sequentially consistent.
    pub fn atomic_load(src: &int) -> int;
    /// Atomic load, acquire ordering.
    pub fn atomic_load_acq(src: &int) -> int;

    /// Atomic store, sequentially consistent.
    pub fn atomic_store(dst: &mut int, val: int);
    /// Atomic store, release ordering.
    pub fn atomic_store_rel(dst: &mut int, val: int);

    /// Atomic exchange, sequentially consistent.
    pub fn atomic_xchg(dst: &mut int, src: int) -> int;
    /// Atomic exchange, acquire ordering.
    pub fn atomic_xchg_acq(dst: &mut int, src: int) -> int;
    /// Atomic exchange, release ordering.
    pub fn atomic_xchg_rel(dst: &mut int, src: int) -> int;

    /// Atomic addition, sequentially consistent.
    pub fn atomic_xadd(dst: &mut int, src: int) -> int;
    /// Atomic addition, acquire ordering.
    pub fn atomic_xadd_acq(dst: &mut int, src: int) -> int;
    /// Atomic addition, release ordering.
    pub fn atomic_xadd_rel(dst: &mut int, src: int) -> int;

    /// Atomic subtraction, sequentially consistent.
    pub fn atomic_xsub(dst: &mut int, src: int) -> int;
    /// Atomic subtraction, acquire ordering.
    pub fn atomic_xsub_acq(dst: &mut int, src: int) -> int;
    /// Atomic subtraction, release ordering.
    pub fn atomic_xsub_rel(dst: &mut int, src: int) -> int;

    /// The size of a type in bytes.
    ///
    /// This is the exact number of bytes in memory taken up by a
    /// value of the given type. In other words, a memset of this size
    /// would *exactly* overwrite a value. When laid out in vectors
    /// and structures there may be additional padding between
    /// elements.
    pub fn size_of<T>() -> uint;

    /// Move a value to a memory location containing a value.
    ///
    /// Drop glue is run on the destination, which must contain a
    /// valid Rust value.
    pub fn move_val<T>(dst: &mut T, src: T);

    /// Move a value to an uninitialized memory location.
    ///
    /// Drop glue is not run on the destination.
    pub fn move_val_init<T>(dst: &mut T, src: T);

    pub fn min_align_of<T>() -> uint;
    pub fn pref_align_of<T>() -> uint;

    /// Get a static pointer to a type descriptor.
    pub fn get_tydesc<T>() -> *();

    /// Create a value initialized to zero.
    ///
    /// `init` is unsafe because it returns a zeroed-out datum,
    /// which is unsafe unless T is POD. We don't have a POD
    /// kind yet. (See #4074).
    pub unsafe fn init<T>() -> T;

    /// Create an uninitialized value.
    pub unsafe fn uninit<T>() -> T;

    /// Move a value out of scope without running drop glue.
    ///
    /// `forget` is unsafe because the caller is responsible for
    /// ensuring the argument is deallocated already.
    pub unsafe fn forget<T>(_: T) -> ();

    /// Returns `true` if a type requires drop glue.
    pub fn needs_drop<T>() -> bool;

    // XXX: intrinsic uses legacy modes and has reference to TyDesc
    // and TyVisitor which are in librustc
    //fn visit_tydesc(++td: *TyDesc, &&tv: TyVisitor) -> ();
    // XXX: intrinsic uses legacy modes
    //fn frame_address(f: &once fn(*u8));

    /// Get the address of the `__morestack` stack growth function.
    pub fn morestack_addr() -> *();

    /// Equivalent to the `llvm.memmove.p0i8.0i8.i32` intrinsic.
    pub fn memmove32(dst: *mut u8, src: *u8, size: u32);
    /// Equivalent to the `llvm.memmove.p0i8.0i8.i64` intrinsic.
    pub fn memmove64(dst: *mut u8, src: *u8, size: u64);

    pub fn sqrtf32(x: f32) -> f32;
    pub fn sqrtf64(x: f64) -> f64;

    pub fn powif32(a: f32, x: i32) -> f32;
    pub fn powif64(a: f64, x: i32) -> f64;

    // the following kill the stack canary without
    // `fixed_stack_segment`. This possibly only affects the f64
    // variants, but it's hard to be sure since it seems to only
    // occur with fairly specific arguments.
    #[fixed_stack_segment]
    pub fn sinf32(x: f32) -> f32;
    #[fixed_stack_segment]
    pub fn sinf64(x: f64) -> f64;

    #[fixed_stack_segment]
    pub fn cosf32(x: f32) -> f32;
    #[fixed_stack_segment]
    pub fn cosf64(x: f64) -> f64;

    #[fixed_stack_segment]
    pub fn powf32(a: f32, x: f32) -> f32;
    #[fixed_stack_segment]
    pub fn powf64(a: f64, x: f64) -> f64;

    #[fixed_stack_segment]
    pub fn expf32(x: f32) -> f32;
    #[fixed_stack_segment]
    pub fn expf64(x: f64) -> f64;

    pub fn exp2f32(x: f32) -> f32;
    pub fn exp2f64(x: f64) -> f64;

    pub fn logf32(x: f32) -> f32;
    pub fn logf64(x: f64) -> f64;

    pub fn log10f32(x: f32) -> f32;
    pub fn log10f64(x: f64) -> f64;

    pub fn log2f32(x: f32) -> f32;
    pub fn log2f64(x: f64) -> f64;

    pub fn fmaf32(a: f32, b: f32, c: f32) -> f32;
    pub fn fmaf64(a: f64, b: f64, c: f64) -> f64;

    pub fn fabsf32(x: f32) -> f32;
    pub fn fabsf64(x: f64) -> f64;

    pub fn floorf32(x: f32) -> f32;
    pub fn floorf64(x: f64) -> f64;

    pub fn ceilf32(x: f32) -> f32;
    pub fn ceilf64(x: f64) -> f64;

    pub fn truncf32(x: f32) -> f32;
    pub fn truncf64(x: f64) -> f64;

    pub fn ctpop8(x: i8) -> i8;
    pub fn ctpop16(x: i16) -> i16;
    pub fn ctpop32(x: i32) -> i32;
    pub fn ctpop64(x: i64) -> i64;

    pub fn ctlz8(x: i8) -> i8;
    pub fn ctlz16(x: i16) -> i16;
    pub fn ctlz32(x: i32) -> i32;
    pub fn ctlz64(x: i64) -> i64;

    pub fn cttz8(x: i8) -> i8;
    pub fn cttz16(x: i16) -> i16;
    pub fn cttz32(x: i32) -> i32;
    pub fn cttz64(x: i64) -> i64;

    pub fn bswap16(x: i16) -> i16;
    pub fn bswap32(x: i32) -> i32;
    pub fn bswap64(x: i64) -> i64;
}
