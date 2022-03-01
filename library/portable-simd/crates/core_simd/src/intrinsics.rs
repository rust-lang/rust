//! This module contains the LLVM intrinsics bindings that provide the functionality for this
//! crate.
//!
//! The LLVM assembly language is documented here: <https://llvm.org/docs/LangRef.html>
//!
//! A quick glossary of jargon that may appear in this module, mostly paraphrasing LLVM's LangRef:
//! - poison: "undefined behavior as a value". specifically, it is like uninit memory (such as padding bytes). it is "safe" to create poison, BUT
//!   poison MUST NOT be observed from safe code, as operations on poison return poison, like NaN. unlike NaN, which has defined comparisons,
//!   poison is neither true nor false, and LLVM may also convert it to undef (at which point it is both). so, it can't be conditioned on, either.
//! - undef: "a value that is every value". functionally like poison, insofar as Rust is concerned. poison may become this. note:
//!   this means that division by poison or undef is like division by zero, which means it inflicts...
//! - "UB": poison and undef cover most of what people call "UB". "UB" means this operation immediately invalidates the program:
//!   LLVM is allowed to lower it to `ud2` or other opcodes that may cause an illegal instruction exception, and this is the "good end".
//!   The "bad end" is that LLVM may reverse time to the moment control flow diverged on a path towards undefined behavior,
//!   and destroy the other branch, potentially deleting safe code and violating Rust's `unsafe` contract.
//!
//! Note that according to LLVM, vectors are not arrays, but they are equivalent when stored to and loaded from memory.
//!
//! Unless stated otherwise, all intrinsics for binary operations require SIMD vectors of equal types and lengths.

/// These intrinsics aren't linked directly from LLVM and are mostly undocumented, however they are
/// mostly lowered to the matching LLVM instructions by the compiler in a fairly straightforward manner.
/// The associated LLVM instruction or intrinsic is documented alongside each Rust intrinsic function.
extern "platform-intrinsic" {
    /// add/fadd
    pub(crate) fn simd_add<T>(x: T, y: T) -> T;

    /// sub/fsub
    pub(crate) fn simd_sub<T>(lhs: T, rhs: T) -> T;

    /// mul/fmul
    pub(crate) fn simd_mul<T>(x: T, y: T) -> T;

    /// udiv/sdiv/fdiv
    /// ints and uints: {s,u}div incur UB if division by zero occurs.
    /// ints: sdiv is UB for int::MIN / -1.
    /// floats: fdiv is never UB, but may create NaNs or infinities.
    pub(crate) fn simd_div<T>(lhs: T, rhs: T) -> T;

    /// urem/srem/frem
    /// ints and uints: {s,u}rem incur UB if division by zero occurs.
    /// ints: srem is UB for int::MIN / -1.
    /// floats: frem is equivalent to libm::fmod in the "default" floating point environment, sans errno.
    pub(crate) fn simd_rem<T>(lhs: T, rhs: T) -> T;

    /// shl
    /// for (u)ints. poison if rhs >= lhs::BITS
    pub(crate) fn simd_shl<T>(lhs: T, rhs: T) -> T;

    /// ints: ashr
    /// uints: lshr
    /// poison if rhs >= lhs::BITS
    pub(crate) fn simd_shr<T>(lhs: T, rhs: T) -> T;

    /// and
    pub(crate) fn simd_and<T>(x: T, y: T) -> T;

    /// or
    pub(crate) fn simd_or<T>(x: T, y: T) -> T;

    /// xor
    pub(crate) fn simd_xor<T>(x: T, y: T) -> T;

    /// fptoui/fptosi/uitofp/sitofp
    /// casting floats to integers is truncating, so it is safe to convert values like e.g. 1.5
    /// but the truncated value must fit in the target type or the result is poison.
    /// use `simd_as` instead for a cast that performs a saturating conversion.
    pub(crate) fn simd_cast<T, U>(x: T) -> U;
    /// follows Rust's `T as U` semantics, including saturating float casts
    /// which amounts to the same as `simd_cast` for many cases
    pub(crate) fn simd_as<T, U>(x: T) -> U;

    /// neg/fneg
    /// ints: ultimately becomes a call to cg_ssa's BuilderMethods::neg. cg_llvm equates this to `simd_sub(Simd::splat(0), x)`.
    /// floats: LLVM's fneg, which changes the floating point sign bit. Some arches have instructions for it.
    /// Rust panics for Neg::neg(int::MIN) due to overflow, but it is not UB in LLVM without `nsw`.
    pub(crate) fn simd_neg<T>(x: T) -> T;

    /// fabs
    pub(crate) fn simd_fabs<T>(x: T) -> T;

    // minnum/maxnum
    pub(crate) fn simd_fmin<T>(x: T, y: T) -> T;
    pub(crate) fn simd_fmax<T>(x: T, y: T) -> T;

    // these return Simd<int, N> with the same BITS size as the inputs
    pub(crate) fn simd_eq<T, U>(x: T, y: T) -> U;
    pub(crate) fn simd_ne<T, U>(x: T, y: T) -> U;
    pub(crate) fn simd_lt<T, U>(x: T, y: T) -> U;
    pub(crate) fn simd_le<T, U>(x: T, y: T) -> U;
    pub(crate) fn simd_gt<T, U>(x: T, y: T) -> U;
    pub(crate) fn simd_ge<T, U>(x: T, y: T) -> U;

    // shufflevector
    // idx: LLVM calls it a "shuffle mask vector constant", a vector of i32s
    pub(crate) fn simd_shuffle<T, U, V>(x: T, y: T, idx: U) -> V;

    /// llvm.masked.gather
    /// like a loop of pointer reads
    /// val: vector of values to select if a lane is masked
    /// ptr: vector of pointers to read from
    /// mask: a "wide" mask of integers, selects as if simd_select(mask, read(ptr), val)
    /// note, the LLVM intrinsic accepts a mask vector of <N x i1>
    /// FIXME: review this if/when we fix up our mask story in general?
    pub(crate) fn simd_gather<T, U, V>(val: T, ptr: U, mask: V) -> T;
    /// llvm.masked.scatter
    /// like gather, but more spicy, as it writes instead of reads
    pub(crate) fn simd_scatter<T, U, V>(val: T, ptr: U, mask: V);

    // {s,u}add.sat
    pub(crate) fn simd_saturating_add<T>(x: T, y: T) -> T;

    // {s,u}sub.sat
    pub(crate) fn simd_saturating_sub<T>(lhs: T, rhs: T) -> T;

    // reductions
    // llvm.vector.reduce.{add,fadd}
    pub(crate) fn simd_reduce_add_ordered<T, U>(x: T, y: U) -> U;
    // llvm.vector.reduce.{mul,fmul}
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
    // first argument is a vector of integers, -1 (all bits 1) is "true"
    // logically equivalent to (yes & m) | (no & (m^-1),
    // but you can use it on floats.
    pub(crate) fn simd_select<M, T>(m: M, yes: T, no: T) -> T;
    #[allow(unused)]
    pub(crate) fn simd_select_bitmask<M, T>(m: M, yes: T, no: T) -> T;
}
