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

# Volatiles

The volatile intrinsics provide operations intended to act on I/O
memory, which are guaranteed to not be reordered by the compiler
across other volatile intrinsics. See the LLVM documentation on
[[volatile]].

[volatile]: http://llvm.org/docs/LangRef.html#volatile-memory-accesses

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
pub use realstd::unstable::intrinsics::{TyDesc, Opaque, TyVisitor, TypeId};

pub type GlueFn = extern "Rust" fn(*i8);

#[lang="ty_desc"]
#[cfg(not(test))]
pub struct TyDesc {
    // sizeof(T)
    size: uint,

    // alignof(T)
    align: uint,

    // Called when a value of type `T` is no longer needed
    drop_glue: GlueFn,

    // Called by reflection visitor to visit a value of type `T`
    visit_glue: GlueFn,

    // Name corresponding to the type
    name: &'static str
}

#[lang="opaque"]
#[cfg(not(test))]
pub enum Opaque { }

pub type Disr = u64;

#[lang="ty_visitor"]
#[cfg(not(test))]
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

    fn visit_f32(&mut self) -> bool;
    fn visit_f64(&mut self) -> bool;

    fn visit_char(&mut self) -> bool;

    fn visit_estr_box(&mut self) -> bool;
    fn visit_estr_uniq(&mut self) -> bool;
    fn visit_estr_slice(&mut self) -> bool;
    fn visit_estr_fixed(&mut self, n: uint, sz: uint, align: uint) -> bool;

    fn visit_box(&mut self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_uniq(&mut self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_ptr(&mut self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_rptr(&mut self, mtbl: uint, inner: *TyDesc) -> bool;

    fn visit_vec(&mut self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_unboxed_vec(&mut self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_evec_box(&mut self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_evec_uniq(&mut self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_evec_slice(&mut self, mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_evec_fixed(&mut self, n: uint, sz: uint, align: uint,
                        mtbl: uint, inner: *TyDesc) -> bool;

    fn visit_enter_rec(&mut self, n_fields: uint,
                       sz: uint, align: uint) -> bool;
    fn visit_rec_field(&mut self, i: uint, name: &str,
                       mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_leave_rec(&mut self, n_fields: uint,
                       sz: uint, align: uint) -> bool;

    fn visit_enter_class(&mut self, name: &str, named_fields: bool, n_fields: uint,
                         sz: uint, align: uint) -> bool;
    fn visit_class_field(&mut self, i: uint, name: &str, named: bool,
                         mtbl: uint, inner: *TyDesc) -> bool;
    fn visit_leave_class(&mut self, name: &str, named_fields: bool, n_fields: uint,
                         sz: uint, align: uint) -> bool;

    fn visit_enter_tup(&mut self, n_fields: uint,
                       sz: uint, align: uint) -> bool;
    fn visit_tup_field(&mut self, i: uint, inner: *TyDesc) -> bool;
    fn visit_leave_tup(&mut self, n_fields: uint,
                       sz: uint, align: uint) -> bool;

    fn visit_enter_enum(&mut self, n_variants: uint,
                        get_disr: extern unsafe fn(ptr: *Opaque) -> Disr,
                        sz: uint, align: uint) -> bool;
    fn visit_enter_enum_variant(&mut self, variant: uint,
                                disr_val: Disr,
                                n_fields: uint,
                                name: &str) -> bool;
    fn visit_enum_variant_field(&mut self, i: uint, offset: uint, inner: *TyDesc) -> bool;
    fn visit_leave_enum_variant(&mut self, variant: uint,
                                disr_val: Disr,
                                n_fields: uint,
                                name: &str) -> bool;
    fn visit_leave_enum(&mut self, n_variants: uint,
                        get_disr: extern unsafe fn(ptr: *Opaque) -> Disr,
                        sz: uint, align: uint) -> bool;

    fn visit_enter_fn(&mut self, purity: uint, proto: uint,
                      n_inputs: uint, retstyle: uint) -> bool;
    fn visit_fn_input(&mut self, i: uint, mode: uint, inner: *TyDesc) -> bool;
    fn visit_fn_output(&mut self, retstyle: uint, variadic: bool, inner: *TyDesc) -> bool;
    fn visit_leave_fn(&mut self, purity: uint, proto: uint,
                      n_inputs: uint, retstyle: uint) -> bool;

    fn visit_trait(&mut self, name: &str) -> bool;
    fn visit_param(&mut self, i: uint) -> bool;
    fn visit_self(&mut self) -> bool;
    fn visit_type(&mut self) -> bool;
}

extern "rust-intrinsic" {
    pub fn atomic_cxchg<T>(dst: &mut T, old: T, src: T) -> T;
    pub fn atomic_cxchg_acq<T>(dst: &mut T, old: T, src: T) -> T;
    pub fn atomic_cxchg_rel<T>(dst: &mut T, old: T, src: T) -> T;
    pub fn atomic_cxchg_acqrel<T>(dst: &mut T, old: T, src: T) -> T;
    pub fn atomic_cxchg_relaxed<T>(dst: &mut T, old: T, src: T) -> T;

    pub fn atomic_load<T>(src: &T) -> T;
    pub fn atomic_load_acq<T>(src: &T) -> T;
    pub fn atomic_load_relaxed<T>(src: &T) -> T;

    pub fn atomic_store<T>(dst: &mut T, val: T);
    pub fn atomic_store_rel<T>(dst: &mut T, val: T);
    pub fn atomic_store_relaxed<T>(dst: &mut T, val: T);

    pub fn atomic_xchg<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_xchg_acq<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_xchg_rel<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_xchg_acqrel<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_xchg_relaxed<T>(dst: &mut T, src: T) -> T;

    pub fn atomic_xadd<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_xadd_acq<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_xadd_rel<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_xadd_acqrel<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_xadd_relaxed<T>(dst: &mut T, src: T) -> T;

    pub fn atomic_xsub<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_xsub_acq<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_xsub_rel<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_xsub_acqrel<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_xsub_relaxed<T>(dst: &mut T, src: T) -> T;

    pub fn atomic_and<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_and_acq<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_and_rel<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_and_acqrel<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_and_relaxed<T>(dst: &mut T, src: T) -> T;

    pub fn atomic_nand<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_nand_acq<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_nand_rel<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_nand_acqrel<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_nand_relaxed<T>(dst: &mut T, src: T) -> T;

    pub fn atomic_or<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_or_acq<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_or_rel<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_or_acqrel<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_or_relaxed<T>(dst: &mut T, src: T) -> T;

    pub fn atomic_xor<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_xor_acq<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_xor_rel<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_xor_acqrel<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_xor_relaxed<T>(dst: &mut T, src: T) -> T;

    pub fn atomic_max<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_max_acq<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_max_rel<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_max_acqrel<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_max_relaxed<T>(dst: &mut T, src: T) -> T;

    pub fn atomic_min<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_min_acq<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_min_rel<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_min_acqrel<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_min_relaxed<T>(dst: &mut T, src: T) -> T;

    pub fn atomic_umin<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_umin_acq<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_umin_rel<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_umin_acqrel<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_umin_relaxed<T>(dst: &mut T, src: T) -> T;

    pub fn atomic_umax<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_umax_acq<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_umax_rel<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_umax_acqrel<T>(dst: &mut T, src: T) -> T;
    pub fn atomic_umax_relaxed<T>(dst: &mut T, src: T) -> T;

    pub fn atomic_fence();
    pub fn atomic_fence_acq();
    pub fn atomic_fence_rel();
    pub fn atomic_fence_acqrel();

    /// Abort the execution of the process.
    pub fn abort() -> !;

    /// Execute a breakpoint trap, for inspection by a debugger.
    pub fn breakpoint();

    pub fn volatile_load<T>(src: *T) -> T;
    pub fn volatile_store<T>(dst: *mut T, val: T);


    /// The size of a type in bytes.
    ///
    /// This is the exact number of bytes in memory taken up by a
    /// value of the given type. In other words, a memset of this size
    /// would *exactly* overwrite a value. When laid out in vectors
    /// and structures there may be additional padding between
    /// elements.
    pub fn size_of<T>() -> uint;

    /// Move a value to an uninitialized memory location.
    ///
    /// Drop glue is not run on the destination.
    pub fn move_val_init<T>(dst: &mut T, src: T);

    pub fn min_align_of<T>() -> uint;
    pub fn pref_align_of<T>() -> uint;

    /// Get a static pointer to a type descriptor.
    pub fn get_tydesc<T>() -> *TyDesc;

    /// Gets an identifier which is globally unique to the specified type. This
    /// function will return the same value for a type regardless of whichever
    /// crate it is invoked in.
    pub fn type_id<T: 'static>() -> TypeId;


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
    pub fn owns_managed<T>() -> bool;

    pub fn visit_tydesc(td: *TyDesc, tv: &mut TyVisitor);

    /// Get the address of the `__morestack` stack growth function.
    pub fn morestack_addr() -> *();

    /// Calculates the offset from a pointer. The offset *must* be in-bounds of
    /// the object, or one-byte-past-the-end. An arithmetic overflow is also
    /// undefined behaviour.
    ///
    /// This is implemented as an intrinsic to avoid converting to and from an
    /// integer, since the conversion would throw away aliasing information.
    pub fn offset<T>(dst: *T, offset: int) -> *T;

    /// Equivalent to the appropriate `llvm.memcpy.p0i8.0i8.*` intrinsic, with
    /// a size of `count` * `size_of::<T>()` and an alignment of
    /// `min_align_of::<T>()`
    pub fn copy_nonoverlapping_memory<T>(dst: *mut T, src: *T, count: uint);

    /// Equivalent to the appropriate `llvm.memmove.p0i8.0i8.*` intrinsic, with
    /// a size of `count` * `size_of::<T>()` and an alignment of
    /// `min_align_of::<T>()`
    pub fn copy_memory<T>(dst: *mut T, src: *T, count: uint);

    /// Equivalent to the appropriate `llvm.memset.p0i8.*` intrinsic, with a
    /// size of `count` * `size_of::<T>()` and an alignment of
    /// `min_align_of::<T>()`
    pub fn set_memory<T>(dst: *mut T, val: u8, count: uint);

    pub fn sqrtf32(x: f32) -> f32;
    pub fn sqrtf64(x: f64) -> f64;

    pub fn powif32(a: f32, x: i32) -> f32;
    pub fn powif64(a: f64, x: i32) -> f64;

    pub fn sinf32(x: f32) -> f32;
    pub fn sinf64(x: f64) -> f64;

    pub fn cosf32(x: f32) -> f32;
    pub fn cosf64(x: f64) -> f64;

    pub fn powf32(a: f32, x: f32) -> f32;
    pub fn powf64(a: f64, x: f64) -> f64;

    pub fn expf32(x: f32) -> f32;
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

    pub fn copysignf32(x: f32, y: f32) -> f32;
    pub fn copysignf64(x: f64, y: f64) -> f64;

    pub fn floorf32(x: f32) -> f32;
    pub fn floorf64(x: f64) -> f64;

    pub fn ceilf32(x: f32) -> f32;
    pub fn ceilf64(x: f64) -> f64;

    pub fn truncf32(x: f32) -> f32;
    pub fn truncf64(x: f64) -> f64;

    pub fn rintf32(x: f32) -> f32;
    pub fn rintf64(x: f64) -> f64;

    pub fn nearbyintf32(x: f32) -> f32;
    pub fn nearbyintf64(x: f64) -> f64;

    pub fn roundf32(x: f32) -> f32;
    pub fn roundf64(x: f64) -> f64;

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

#[cfg(target_endian = "little")] #[inline] pub fn to_le16(x: i16) -> i16 { x }
#[cfg(target_endian = "big")]    #[inline] pub fn to_le16(x: i16) -> i16 { unsafe { bswap16(x) } }
#[cfg(target_endian = "little")] #[inline] pub fn to_le32(x: i32) -> i32 { x }
#[cfg(target_endian = "big")]    #[inline] pub fn to_le32(x: i32) -> i32 { unsafe { bswap32(x) } }
#[cfg(target_endian = "little")] #[inline] pub fn to_le64(x: i64) -> i64 { x }
#[cfg(target_endian = "big")]    #[inline] pub fn to_le64(x: i64) -> i64 { unsafe { bswap64(x) } }

#[cfg(target_endian = "little")] #[inline] pub fn to_be16(x: i16) -> i16 { unsafe { bswap16(x) } }
#[cfg(target_endian = "big")]    #[inline] pub fn to_be16(x: i16) -> i16 { x }
#[cfg(target_endian = "little")] #[inline] pub fn to_be32(x: i32) -> i32 { unsafe { bswap32(x) } }
#[cfg(target_endian = "big")]    #[inline] pub fn to_be32(x: i32) -> i32 { x }
#[cfg(target_endian = "little")] #[inline] pub fn to_be64(x: i64) -> i64 { unsafe { bswap64(x) } }
#[cfg(target_endian = "big")]    #[inline] pub fn to_be64(x: i64) -> i64 { x }

#[cfg(target_endian = "little")] #[inline] pub fn from_le16(x: i16) -> i16 { x }
#[cfg(target_endian = "big")]    #[inline] pub fn from_le16(x: i16) -> i16 { unsafe { bswap16(x) } }
#[cfg(target_endian = "little")] #[inline] pub fn from_le32(x: i32) -> i32 { x }
#[cfg(target_endian = "big")]    #[inline] pub fn from_le32(x: i32) -> i32 { unsafe { bswap32(x) } }
#[cfg(target_endian = "little")] #[inline] pub fn from_le64(x: i64) -> i64 { x }
#[cfg(target_endian = "big")]    #[inline] pub fn from_le64(x: i64) -> i64 { unsafe { bswap64(x) } }

#[cfg(target_endian = "little")] #[inline] pub fn from_be16(x: i16) -> i16 { unsafe { bswap16(x) } }
#[cfg(target_endian = "big")]    #[inline] pub fn from_be16(x: i16) -> i16 { x }
#[cfg(target_endian = "little")] #[inline] pub fn from_be32(x: i32) -> i32 { unsafe { bswap32(x) } }
#[cfg(target_endian = "big")]    #[inline] pub fn from_be32(x: i32) -> i32 { x }
#[cfg(target_endian = "little")] #[inline] pub fn from_be64(x: i64) -> i64 { unsafe { bswap64(x) } }
#[cfg(target_endian = "big")]    #[inline] pub fn from_be64(x: i64) -> i64 { x }

/// `TypeId` represents a globally unique identifier for a type
#[lang="type_id"] // This needs to be kept in lockstep with the code in trans/intrinsic.rs and
                  // middle/lang_items.rs
#[deriving(Eq, IterBytes)]
#[cfg(not(test))]
pub struct TypeId {
    priv t: u64,
}

#[cfg(not(test))]
impl TypeId {
    /// Returns the `TypeId` of the type this generic function has been instantiated with
    pub fn of<T: 'static>() -> TypeId {
        unsafe { type_id::<T>() }
    }
}
