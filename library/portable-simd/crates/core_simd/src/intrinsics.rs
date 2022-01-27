//! This module contains the LLVM intrinsics bindings that provide the functionality for this
//! crate.
//!
//! The LLVM assembly language is documented here: <https://llvm.org/docs/LangRef.html>

/// These intrinsics aren't linked directly from LLVM and are mostly undocumented, however they are
/// simply lowered to the matching LLVM instructions by the compiler.  The associated instruction
/// is documented alongside each intrinsic.
extern "platform-intrinsic" {
    /// add/fadd
    pub(crate) fn simd_add<T>(x: T, y: T) -> T;

    /// sub/fsub
    pub(crate) fn simd_sub<T>(x: T, y: T) -> T;

    /// mul/fmul
    pub(crate) fn simd_mul<T>(x: T, y: T) -> T;

    /// udiv/sdiv/fdiv
    pub(crate) fn simd_div<T>(x: T, y: T) -> T;

    /// urem/srem/frem
    pub(crate) fn simd_rem<T>(x: T, y: T) -> T;

    /// shl
    pub(crate) fn simd_shl<T>(x: T, y: T) -> T;

    /// lshr/ashr
    pub(crate) fn simd_shr<T>(x: T, y: T) -> T;

    /// and
    pub(crate) fn simd_and<T>(x: T, y: T) -> T;

    /// or
    pub(crate) fn simd_or<T>(x: T, y: T) -> T;

    /// xor
    pub(crate) fn simd_xor<T>(x: T, y: T) -> T;

    /// fptoui/fptosi/uitofp/sitofp
    pub(crate) fn simd_cast<T, U>(x: T) -> U;
    /// follows Rust's `T as U` semantics, including saturating float casts
    /// which amounts to the same as `simd_cast` for many cases
    #[cfg(not(bootstrap))]
    pub(crate) fn simd_as<T, U>(x: T) -> U;

    /// neg/fneg
    pub(crate) fn simd_neg<T>(x: T) -> T;

    /// fabs
    pub(crate) fn simd_fabs<T>(x: T) -> T;

    // minnum/maxnum
    pub(crate) fn simd_fmin<T>(x: T, y: T) -> T;
    pub(crate) fn simd_fmax<T>(x: T, y: T) -> T;

    pub(crate) fn simd_eq<T, U>(x: T, y: T) -> U;
    pub(crate) fn simd_ne<T, U>(x: T, y: T) -> U;
    pub(crate) fn simd_lt<T, U>(x: T, y: T) -> U;
    pub(crate) fn simd_le<T, U>(x: T, y: T) -> U;
    pub(crate) fn simd_gt<T, U>(x: T, y: T) -> U;
    pub(crate) fn simd_ge<T, U>(x: T, y: T) -> U;

    // shufflevector
    pub(crate) fn simd_shuffle<T, U, V>(x: T, y: T, idx: U) -> V;

    pub(crate) fn simd_gather<T, U, V>(val: T, ptr: U, mask: V) -> T;
    pub(crate) fn simd_scatter<T, U, V>(val: T, ptr: U, mask: V);

    // {s,u}add.sat
    pub(crate) fn simd_saturating_add<T>(x: T, y: T) -> T;

    // {s,u}sub.sat
    pub(crate) fn simd_saturating_sub<T>(x: T, y: T) -> T;

    // reductions
    pub(crate) fn simd_reduce_add_ordered<T, U>(x: T, y: U) -> U;
    pub(crate) fn simd_reduce_mul_ordered<T, U>(x: T, y: U) -> U;
    #[allow(unused)]
    pub(crate) fn simd_reduce_all<T>(x: T) -> bool;
    #[allow(unused)]
    pub(crate) fn simd_reduce_any<T>(x: T) -> bool;
    pub(crate) fn simd_reduce_max<T, U>(x: T) -> U;
    pub(crate) fn simd_reduce_min<T, U>(x: T) -> U;
    pub(crate) fn simd_reduce_and<T, U>(x: T) -> U;
    pub(crate) fn simd_reduce_or<T, U>(x: T) -> U;
    pub(crate) fn simd_reduce_xor<T, U>(x: T) -> U;

    // truncate integer vector to bitmask
    #[allow(unused)]
    pub(crate) fn simd_bitmask<T, U>(x: T) -> U;

    // select
    pub(crate) fn simd_select<M, T>(m: M, a: T, b: T) -> T;
    #[allow(unused)]
    pub(crate) fn simd_select_bitmask<M, T>(m: M, a: T, b: T) -> T;
}
