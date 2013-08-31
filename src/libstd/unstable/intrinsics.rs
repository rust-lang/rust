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
semantics as C++11. See the LLVM documentation on [[atomics]].

[atomics]: http://llvm.org/docs/Atomics.html

A quick refresher on memory ordering:

* Acquire - a barrier for acquiring a lock. Subsequent reads and writes
  take place after the barrier.
* Release - a barrier for releasing a lock. Preceding reads and writes
  take place before the barrier.
* Sequentially consistent - sequentially consistent operations are
  guaranteed to happen in order. This is the standard mode for working
  with atomic types and is equivalent to Java's `volatile`.

*/

// This is needed to prevent duplicate lang item definitions.
#[cfg(test)]
pub use realstd::unstable::intrinsics::{TyDesc, Opaque, TyVisitor};

pub type GlueFn = extern "Rust" fn(*i8);

// NB: this has to be kept in sync with `type_desc` in `rt`
#[lang="ty_desc"]
#[cfg(not(test))]
pub struct TyDesc {
    // sizeof(T)
    size: uint,

    // alignof(T)
    align: uint,

    // Called on a copy of a value of type `T` *after* memcpy
    take_glue: GlueFn,

    // Called when a value of type `T` is no longer needed
    drop_glue: GlueFn,

    // Called by drop glue when a value of type `T` can be freed
    free_glue: GlueFn,

    // Called by reflection visitor to visit a value of type `T`
    visit_glue: GlueFn,

    // If T represents a box pointer (`@U` or `~U`), then
    // `borrow_offset` is the amount that the pointer must be adjusted
    // to find the payload.  This is always derivable from the type
    // `U`, but in the case of `@Trait` or `~Trait` objects, the type
    // `U` is unknown.
    borrow_offset: uint,
}

#[lang="opaque"]
#[cfg(not(test))]
pub enum Opaque { }

#[lang="ty_visitor"]
#[cfg(not(test), stage0)]
pub trait TyVisitor {
    fn visit_bot(&self) -> bool;
    fn visit_nil(&self) -> bool;
    fn visit_bool(&self) -> bool;

    fn visit_int(&self) -> bool;
    fn visit_i8(&self) -> bool;
    fn visit_i16(&self) -> bool;
    fn visit_i32(&self) -> bool;
    fn visit_i64(&self) -> bool;

    fn visit_uint(&self) -> bool;
    fn visit_u8(&self) -> bool;
    fn visit_u16(&self) -> bool;
    fn visit_u32(&self) -> bool;
    fn visit_u64(&self) -> bool;

    fn visit_float(&self) -> bool;
    fn visit_f32(&self) -> bool;
    fn visit_f64(&self) -> bool;

    fn visit_char(&self) -> bool;

    fn visit_estr_box(&self) -> bool;
    fn visit_estr_uniq(&self) -> bool;
    fn visit_estr_slice(&self) -> bool;
    fn visit_estr_fixed(&self, n: uint, sz: uint, align: uint) -> bool;

    fn visit_box(&self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_uniq(&self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_uniq_managed(&self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_ptr(&self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_rptr(&self, mtbl: uint, inner: *TyDesc) -> bool;

    fn visit_vec(&self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_unboxed_vec(&self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_evec_box(&self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_evec_uniq(&self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_evec_uniq_managed(&self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_evec_slice(&self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_evec_fixed(&self, n: uint, sz: uint, align: uint,
                        mtbl: uint, inner: *TyDesc) -> bool;

    fn visit_enter_rec(&self, n_fields: uint,
                       sz: uint, align: uint) -> bool;
    fn visit_rec_field(&self, i: uint, name: &str,
                       mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_leave_rec(&self, n_fields: uint,
                       sz: uint, align: uint) -> bool;

    fn visit_enter_class(&self, n_fields: uint,
                         sz: uint, align: uint) -> bool;
    fn visit_class_field(&self, i: uint, name: &str,
                         mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_leave_class(&self, n_fields: uint,
                         sz: uint, align: uint) -> bool;

    fn visit_enter_tup(&self, n_fields: uint,
                       sz: uint, align: uint) -> bool;
    fn visit_tup_field(&self, i: uint, inner: *TyDesc) -> bool;
    fn visit_leave_tup(&self, n_fields: uint,
                       sz: uint, align: uint) -> bool;

    fn visit_enter_enum(&self, n_variants: uint,
                        get_disr: extern unsafe fn(ptr: *Opaque) -> int,
                        sz: uint, align: uint) -> bool;
    fn visit_enter_enum_variant(&self, variant: uint,
                                disr_val: int,
                                n_fields: uint,
                                name: &str) -> bool;
    fn visit_enum_variant_field(&self, i: uint, offset: uint, inner: *TyDesc) -> bool;
    fn visit_leave_enum_variant(&self, variant: uint,
                                disr_val: int,
                                n_fields: uint,
                                name: &str) -> bool;
    fn visit_leave_enum(&self, n_variants: uint,
                        get_disr: extern unsafe fn(ptr: *Opaque) -> int,
                        sz: uint, align: uint) -> bool;

    fn visit_enter_fn(&self, purity: uint, proto: uint,
                      n_inputs: uint, retstyle: uint) -> bool;
    fn visit_fn_input(&self, i: uint, mode: uint, inner: *TyDesc) -> bool;
    fn visit_fn_output(&self, retstyle: uint, inner: *TyDesc) -> bool;
    fn visit_leave_fn(&self, purity: uint, proto: uint,
                      n_inputs: uint, retstyle: uint) -> bool;

    fn visit_trait(&self) -> bool;
    fn visit_param(&self, i: uint) -> bool;
    fn visit_self(&self) -> bool;
    fn visit_type(&self) -> bool;
    fn visit_opaque_box(&self) -> bool;
    fn visit_closure_ptr(&self, ck: uint) -> bool;
}

#[lang="ty_visitor"]
#[cfg(not(test), not(stage0))]
pub trait TyVisitor {
    fn visit_bot(&mut self) -> bool;
    fn visit_nil(&mut self) -> bool;
    fn visit_bool(&mut self) -> bool;

    fn visit_int(&mut self) -> bool;
    fn visit_i8(&mut self) -> bool;
    fn visit_i16(&mut self) -> bool;
    fn visit_i32(&mut self) -> bool;
    fn visit_i64(&mut self) -> bool;

    fn visit_uint(&mut self) -> bool;
    fn visit_u8(&mut self) -> bool;
    fn visit_u16(&mut self) -> bool;
    fn visit_u32(&mut self) -> bool;
    fn visit_u64(&mut self) -> bool;

    fn visit_float(&mut self) -> bool;
    fn visit_f32(&mut self) -> bool;
    fn visit_f64(&mut self) -> bool;

    fn visit_char(&mut self) -> bool;

    fn visit_estr_box(&mut self) -> bool;
    fn visit_estr_uniq(&mut self) -> bool;
    fn visit_estr_slice(&mut self) -> bool;
    fn visit_estr_fixed(&mut self, n: uint, sz: uint, align: uint) -> bool;

    fn visit_box(&mut self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_uniq(&mut self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_uniq_managed(&mut self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_ptr(&mut self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_rptr(&mut self, mtbl: uint, inner: *TyDesc) -> bool;

    fn visit_vec(&mut self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_unboxed_vec(&mut self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_evec_box(&mut self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_evec_uniq(&mut self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_evec_uniq_managed(&mut self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_evec_slice(&mut self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_evec_fixed(&mut self, n: uint, sz: uint, align: uint,
                        mtbl: uint, inner: *TyDesc) -> bool;

    fn visit_enter_rec(&mut self, n_fields: uint,
                       sz: uint, align: uint) -> bool;
    fn visit_rec_field(&mut self, i: uint, name: &str,
                       mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_leave_rec(&mut self, n_fields: uint,
                       sz: uint, align: uint) -> bool;

    fn visit_enter_class(&mut self, name: &str, n_fields: uint,
                         sz: uint, align: uint) -> bool;
    fn visit_class_field(&mut self, i: uint, name: &str,
                         mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_leave_class(&mut self, name: &str, n_fields: uint,
                         sz: uint, align: uint) -> bool;

    fn visit_enter_tup(&mut self, n_fields: uint,
                       sz: uint, align: uint) -> bool;
    fn visit_tup_field(&mut self, i: uint, inner: *TyDesc) -> bool;
    fn visit_leave_tup(&mut self, n_fields: uint,
                       sz: uint, align: uint) -> bool;

    fn visit_enter_enum(&mut self, n_variants: uint,
                        get_disr: extern unsafe fn(ptr: *Opaque) -> int,
                        sz: uint, align: uint) -> bool;
    fn visit_enter_enum_variant(&mut self, variant: uint,
                                disr_val: int,
                                n_fields: uint,
                                name: &str) -> bool;
    fn visit_enum_variant_field(&mut self, i: uint, offset: uint, inner: *TyDesc) -> bool;
    fn visit_leave_enum_variant(&mut self, variant: uint,
                                disr_val: int,
                                n_fields: uint,
                                name: &str) -> bool;
    fn visit_leave_enum(&mut self, n_variants: uint,
                        get_disr: extern unsafe fn(ptr: *Opaque) -> int,
                        sz: uint, align: uint) -> bool;

    fn visit_enter_fn(&mut self, purity: uint, proto: uint,
                      n_inputs: uint, retstyle: uint) -> bool;
    fn visit_fn_input(&mut self, i: uint, mode: uint, inner: *TyDesc) -> bool;
    fn visit_fn_output(&mut self, retstyle: uint, inner: *TyDesc) -> bool;
    fn visit_leave_fn(&mut self, purity: uint, proto: uint,
                      n_inputs: uint, retstyle: uint) -> bool;

    fn visit_trait(&mut self) -> bool;
    fn visit_param(&mut self, i: uint) -> bool;
    fn visit_self(&mut self) -> bool;
    fn visit_type(&mut self) -> bool;
    fn visit_opaque_box(&mut self) -> bool;
    fn visit_closure_ptr(&mut self, ck: uint) -> bool;
}

#[abi = "rust-intrinsic"]
extern "rust-intrinsic" {

    /// Atomic compare and exchange, sequentially consistent.
    pub fn atomic_cxchg(dst: &mut int, old: int, src: int) -> int;
    /// Atomic compare and exchange, acquire ordering.
    pub fn atomic_cxchg_acq(dst: &mut int, old: int, src: int) -> int;
    /// Atomic compare and exchange, release ordering.
    pub fn atomic_cxchg_rel(dst: &mut int, old: int, src: int) -> int;

    pub fn atomic_cxchg_acqrel(dst: &mut int, old: int, src: int) -> int;
    pub fn atomic_cxchg_relaxed(dst: &mut int, old: int, src: int) -> int;


    /// Atomic load, sequentially consistent.
    pub fn atomic_load(src: &int) -> int;
    /// Atomic load, acquire ordering.
    pub fn atomic_load_acq(src: &int) -> int;

    pub fn atomic_load_relaxed(src: &int) -> int;

    /// Atomic store, sequentially consistent.
    pub fn atomic_store(dst: &mut int, val: int);
    /// Atomic store, release ordering.
    pub fn atomic_store_rel(dst: &mut int, val: int);

    pub fn atomic_store_relaxed(dst: &mut int, val: int);

    /// Atomic exchange, sequentially consistent.
    pub fn atomic_xchg(dst: &mut int, src: int) -> int;
    /// Atomic exchange, acquire ordering.
    pub fn atomic_xchg_acq(dst: &mut int, src: int) -> int;
    /// Atomic exchange, release ordering.
    pub fn atomic_xchg_rel(dst: &mut int, src: int) -> int;
    pub fn atomic_xchg_acqrel(dst: &mut int, src: int) -> int;
    pub fn atomic_xchg_relaxed(dst: &mut int, src: int) -> int;

    /// Atomic addition, sequentially consistent.
    pub fn atomic_xadd(dst: &mut int, src: int) -> int;
    /// Atomic addition, acquire ordering.
    pub fn atomic_xadd_acq(dst: &mut int, src: int) -> int;
    /// Atomic addition, release ordering.
    pub fn atomic_xadd_rel(dst: &mut int, src: int) -> int;
    pub fn atomic_xadd_acqrel(dst: &mut int, src: int) -> int;
    pub fn atomic_xadd_relaxed(dst: &mut int, src: int) -> int;

    /// Atomic subtraction, sequentially consistent.
    pub fn atomic_xsub(dst: &mut int, src: int) -> int;
    /// Atomic subtraction, acquire ordering.
    pub fn atomic_xsub_acq(dst: &mut int, src: int) -> int;
    /// Atomic subtraction, release ordering.
    pub fn atomic_xsub_rel(dst: &mut int, src: int) -> int;
    pub fn atomic_xsub_acqrel(dst: &mut int, src: int) -> int;
    pub fn atomic_xsub_relaxed(dst: &mut int, src: int) -> int;

    pub fn atomic_and(dst: &mut int, src: int) -> int;
    pub fn atomic_and_acq(dst: &mut int, src: int) -> int;
    pub fn atomic_and_rel(dst: &mut int, src: int) -> int;
    pub fn atomic_and_acqrel(dst: &mut int, src: int) -> int;
    pub fn atomic_and_relaxed(dst: &mut int, src: int) -> int;

    pub fn atomic_nand(dst: &mut int, src: int) -> int;
    pub fn atomic_nand_acq(dst: &mut int, src: int) -> int;
    pub fn atomic_nand_rel(dst: &mut int, src: int) -> int;
    pub fn atomic_nand_acqrel(dst: &mut int, src: int) -> int;
    pub fn atomic_nand_relaxed(dst: &mut int, src: int) -> int;

    pub fn atomic_or(dst: &mut int, src: int) -> int;
    pub fn atomic_or_acq(dst: &mut int, src: int) -> int;
    pub fn atomic_or_rel(dst: &mut int, src: int) -> int;
    pub fn atomic_or_acqrel(dst: &mut int, src: int) -> int;
    pub fn atomic_or_relaxed(dst: &mut int, src: int) -> int;

    pub fn atomic_xor(dst: &mut int, src: int) -> int;
    pub fn atomic_xor_acq(dst: &mut int, src: int) -> int;
    pub fn atomic_xor_rel(dst: &mut int, src: int) -> int;
    pub fn atomic_xor_acqrel(dst: &mut int, src: int) -> int;
    pub fn atomic_xor_relaxed(dst: &mut int, src: int) -> int;

    pub fn atomic_max(dst: &mut int, src: int) -> int;
    pub fn atomic_max_acq(dst: &mut int, src: int) -> int;
    pub fn atomic_max_rel(dst: &mut int, src: int) -> int;
    pub fn atomic_max_acqrel(dst: &mut int, src: int) -> int;
    pub fn atomic_max_relaxed(dst: &mut int, src: int) -> int;

    pub fn atomic_min(dst: &mut int, src: int) -> int;
    pub fn atomic_min_acq(dst: &mut int, src: int) -> int;
    pub fn atomic_min_rel(dst: &mut int, src: int) -> int;
    pub fn atomic_min_acqrel(dst: &mut int, src: int) -> int;
    pub fn atomic_min_relaxed(dst: &mut int, src: int) -> int;

    pub fn atomic_umin(dst: &mut int, src: int) -> int;
    pub fn atomic_umin_acq(dst: &mut int, src: int) -> int;
    pub fn atomic_umin_rel(dst: &mut int, src: int) -> int;
    pub fn atomic_umin_acqrel(dst: &mut int, src: int) -> int;
    pub fn atomic_umin_relaxed(dst: &mut int, src: int) -> int;

    pub fn atomic_umax(dst: &mut int, src: int) -> int;
    pub fn atomic_umax_acq(dst: &mut int, src: int) -> int;
    pub fn atomic_umax_rel(dst: &mut int, src: int) -> int;
    pub fn atomic_umax_acqrel(dst: &mut int, src: int) -> int;
    pub fn atomic_umax_relaxed(dst: &mut int, src: int) -> int;

    pub fn atomic_fence();
    pub fn atomic_fence_acq();
    pub fn atomic_fence_rel();
    pub fn atomic_fence_acqrel();

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
    pub fn get_tydesc<T>() -> *TyDesc;

    /// Create a value initialized to zero.
    ///
    /// `init` is unsafe because it returns a zeroed-out datum,
    /// which is unsafe unless T is POD. We don't have a POD
    /// kind yet. (See #4074).
    pub fn init<T>() -> T;

    /// Create an uninitialized value.
    pub fn uninit<T>() -> T;

    /// Move a value out of scope without running drop glue.
    ///
    /// `forget` is unsafe because the caller is responsible for
    /// ensuring the argument is deallocated already.
    pub fn forget<T>(_: T) -> ();
    pub fn transmute<T,U>(e: T) -> U;

    /// Returns `true` if a type requires drop glue.
    pub fn needs_drop<T>() -> bool;

    /// Returns `true` if a type is managed (will be allocated on the local heap)
    pub fn contains_managed<T>() -> bool;

    #[cfg(stage0)]
    pub fn visit_tydesc(td: *TyDesc, tv: &TyVisitor);
    #[cfg(not(stage0))]
    pub fn visit_tydesc(td: *TyDesc, tv: &mut TyVisitor);

    pub fn frame_address(f: &once fn(*u8));

    /// Get the address of the `__morestack` stack growth function.
    pub fn morestack_addr() -> *();

    /// Calculates the offset from a pointer. The offset *must* be in-bounds of
    /// the object, or one-byte-past-the-end. An arithmetic overflow is also
    /// undefined behaviour.
    ///
    /// This is implemented as an intrinsic to avoid converting to and from an
    /// integer, since the conversion would throw away aliasing information.
    pub fn offset<T>(dst: *T, offset: int) -> *T;

    /// Equivalent to the `llvm.memcpy.p0i8.0i8.i32` intrinsic, with a size of
    /// `count` * `size_of::<T>()` and an alignment of `min_align_of::<T>()`
    pub fn memcpy32<T>(dst: *mut T, src: *T, count: u32);
    /// Equivalent to the `llvm.memcpy.p0i8.0i8.i64` intrinsic, with a size of
    /// `count` * `size_of::<T>()` and an alignment of `min_align_of::<T>()`
    pub fn memcpy64<T>(dst: *mut T, src: *T, count: u64);

    /// Equivalent to the `llvm.memmove.p0i8.0i8.i32` intrinsic, with a size of
    /// `count` * `size_of::<T>()` and an alignment of `min_align_of::<T>()`
    pub fn memmove32<T>(dst: *mut T, src: *T, count: u32);
    /// Equivalent to the `llvm.memmove.p0i8.0i8.i64` intrinsic, with a size of
    /// `count` * `size_of::<T>()` and an alignment of `min_align_of::<T>()`
    pub fn memmove64<T>(dst: *mut T, src: *T, count: u64);

    /// Equivalent to the `llvm.memset.p0i8.i32` intrinsic, with a size of
    /// `count` * `size_of::<T>()` and an alignment of `min_align_of::<T>()`
    pub fn memset32<T>(dst: *mut T, val: u8, count: u32);
    /// Equivalent to the `llvm.memset.p0i8.i64` intrinsic, with a size of
    /// `count` * `size_of::<T>()` and an alignment of `min_align_of::<T>()`
    pub fn memset64<T>(dst: *mut T, val: u8, count: u64);

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

    pub fn i8_add_with_overflow(x: i8, y: i8) -> (i8, bool);
    pub fn i16_add_with_overflow(x: i16, y: i16) -> (i16, bool);
    pub fn i32_add_with_overflow(x: i32, y: i32) -> (i32, bool);
    pub fn i64_add_with_overflow(x: i64, y: i64) -> (i64, bool);

    pub fn u8_add_with_overflow(x: u8, y: u8) -> (u8, bool);
    pub fn u16_add_with_overflow(x: u16, y: u16) -> (u16, bool);
    pub fn u32_add_with_overflow(x: u32, y: u32) -> (u32, bool);
    pub fn u64_add_with_overflow(x: u64, y: u64) -> (u64, bool);

    pub fn i8_sub_with_overflow(x: i8, y: i8) -> (i8, bool);
    pub fn i16_sub_with_overflow(x: i16, y: i16) -> (i16, bool);
    pub fn i32_sub_with_overflow(x: i32, y: i32) -> (i32, bool);
    pub fn i64_sub_with_overflow(x: i64, y: i64) -> (i64, bool);

    pub fn u8_sub_with_overflow(x: u8, y: u8) -> (u8, bool);
    pub fn u16_sub_with_overflow(x: u16, y: u16) -> (u16, bool);
    pub fn u32_sub_with_overflow(x: u32, y: u32) -> (u32, bool);
    pub fn u64_sub_with_overflow(x: u64, y: u64) -> (u64, bool);

    pub fn i8_mul_with_overflow(x: i8, y: i8) -> (i8, bool);
    pub fn i16_mul_with_overflow(x: i16, y: i16) -> (i16, bool);
    pub fn i32_mul_with_overflow(x: i32, y: i32) -> (i32, bool);
    pub fn i64_mul_with_overflow(x: i64, y: i64) -> (i64, bool);

    pub fn u8_mul_with_overflow(x: u8, y: u8) -> (u8, bool);
    pub fn u16_mul_with_overflow(x: u16, y: u16) -> (u16, bool);
    pub fn u32_mul_with_overflow(x: u32, y: u32) -> (u32, bool);
    pub fn u64_mul_with_overflow(x: u64, y: u64) -> (u64, bool);
}

#[cfg(target_endian = "little")] pub fn to_le16(x: i16) -> i16 { x }
#[cfg(target_endian = "big")]    pub fn to_le16(x: i16) -> i16 { unsafe { bswap16(x) } }
#[cfg(target_endian = "little")] pub fn to_le32(x: i32) -> i32 { x }
#[cfg(target_endian = "big")]    pub fn to_le32(x: i32) -> i32 { unsafe { bswap32(x) } }
#[cfg(target_endian = "little")] pub fn to_le64(x: i64) -> i64 { x }
#[cfg(target_endian = "big")]    pub fn to_le64(x: i64) -> i64 { unsafe { bswap64(x) } }

#[cfg(target_endian = "little")] pub fn to_be16(x: i16) -> i16 { unsafe { bswap16(x) } }
#[cfg(target_endian = "big")]    pub fn to_be16(x: i16) -> i16 { x }
#[cfg(target_endian = "little")] pub fn to_be32(x: i32) -> i32 { unsafe { bswap32(x) } }
#[cfg(target_endian = "big")]    pub fn to_be32(x: i32) -> i32 { x }
#[cfg(target_endian = "little")] pub fn to_be64(x: i64) -> i64 { unsafe { bswap64(x) } }
#[cfg(target_endian = "big")]    pub fn to_be64(x: i64) -> i64 { x }
