//! Aarch64 targets have two possible implementations for atomics:
//! 1. Load-Locked, Store-Conditional (LL/SC), older and slower.
//! 2. Large System Extensions (LSE), newer and faster.
//! To avoid breaking backwards compat, C toolchains introduced a concept of "outlined atomics",
//! where atomic operations call into the compiler runtime to dispatch between two depending on
//! which is supported on the current CPU.
//! See <https://community.arm.com/arm-community-blogs/b/tools-software-ides-blog/posts/making-the-most-of-the-arm-architecture-in-gcc-10#:~:text=out%20of%20line%20atomics> for more discussion.
//!
//! Currently we only support LL/SC, because LSE requires `getauxval` from libc in order to do runtime detection.
//! Use the `compiler-rt` intrinsics if you want LSE support.
//!
//! Ported from `aarch64/lse.S` in LLVM's compiler-rt.
//!
//! Generate functions for each of the following symbols:
//!  __aarch64_casM_ORDER
//!  __aarch64_swpN_ORDER
//!  __aarch64_ldaddN_ORDER
//!  __aarch64_ldclrN_ORDER
//!  __aarch64_ldeorN_ORDER
//!  __aarch64_ldsetN_ORDER
//! for N = {1, 2, 4, 8}, M = {1, 2, 4, 8, 16}, ORDER = { relax, acq, rel, acq_rel }
//!
//! The original `lse.S` has some truly horrifying code that expects to be compiled multiple times with different constants.
//! We do something similar, but with macro arguments.
#![cfg_attr(feature = "c", allow(unused_macros))] // avoid putting the macros into a submodule

// We don't do runtime dispatch so we don't have to worry about the `__aarch64_have_lse_atomics` global ctor.

/// Translate a byte size to a Rust type.
#[rustfmt::skip]
macro_rules! int_ty {
    (1) => { i8 };
    (2) => { i16 };
    (4) => { i32 };
    (8) => { i64 };
    (16) => { i128 };
}

/// Given a byte size and a register number, return a register of the appropriate size.
///
/// See <https://developer.arm.com/documentation/102374/0101/Registers-in-AArch64---general-purpose-registers>.
#[rustfmt::skip]
macro_rules! reg {
    (1, $num:literal) => { concat!("w", $num) };
    (2, $num:literal) => { concat!("w", $num) };
    (4, $num:literal) => { concat!("w", $num) };
    (8, $num:literal) => { concat!("x", $num) };
}

/// Given an atomic ordering, translate it to the acquire suffix for the lxdr aarch64 ASM instruction.
#[rustfmt::skip]
macro_rules! acquire {
    (Relaxed) => { "" };
    (Acquire) => { "a" };
    (Release) => { "" };
    (AcqRel) => { "a" };
}

/// Given an atomic ordering, translate it to the release suffix for the stxr aarch64 ASM instruction.
#[rustfmt::skip]
macro_rules! release {
    (Relaxed) => { "" };
    (Acquire) => { "" };
    (Release) => { "l" };
    (AcqRel) => { "l" };
}

/// Given a size in bytes, translate it to the byte suffix for an aarch64 ASM instruction.
#[rustfmt::skip]
macro_rules! size {
    (1) => { "b" };
    (2) => { "h" };
    (4) => { "" };
    (8) => { "" };
    (16) => { "" };
}

/// Given a byte size, translate it to an Unsigned eXTend instruction
/// with the correct semantics.
///
/// See <https://developer.arm.com/documentation/ddi0596/2020-12/Base-Instructions/UXTB--Unsigned-Extend-Byte--an-alias-of-UBFM->
#[rustfmt::skip]
macro_rules! uxt {
    (1) => { "uxtb" };
    (2) => { "uxth" };
    ($_:tt) => { "mov" };
}

/// Given an atomic ordering and byte size, translate it to a LoaD eXclusive Register instruction
/// with the correct semantics.
///
/// See <https://developer.arm.com/documentation/ddi0596/2020-12/Base-Instructions/LDXR--Load-Exclusive-Register->.
macro_rules! ldxr {
    ($ordering:ident, $bytes:tt) => {
        concat!("ld", acquire!($ordering), "xr", size!($bytes))
    };
}

/// Given an atomic ordering and byte size, translate it to a STore eXclusive Register instruction
/// with the correct semantics.
///
/// See <https://developer.arm.com/documentation/ddi0596/2020-12/Base-Instructions/STXR--Store-Exclusive-Register->.
macro_rules! stxr {
    ($ordering:ident, $bytes:tt) => {
        concat!("st", release!($ordering), "xr", size!($bytes))
    };
}

/// Given an atomic ordering and byte size, translate it to a LoaD eXclusive Pair of registers instruction
/// with the correct semantics.
///
/// See <https://developer.arm.com/documentation/ddi0596/2020-12/Base-Instructions/LDXP--Load-Exclusive-Pair-of-Registers->
macro_rules! ldxp {
    ($ordering:ident) => {
        concat!("ld", acquire!($ordering), "xp")
    };
}

/// Given an atomic ordering and byte size, translate it to a STore eXclusive Pair of registers instruction
/// with the correct semantics.
///
/// See <https://developer.arm.com/documentation/ddi0596/2020-12/Base-Instructions/STXP--Store-Exclusive-Pair-of-registers->.
macro_rules! stxp {
    ($ordering:ident) => {
        concat!("st", release!($ordering), "xp")
    };
}

/// See <https://doc.rust-lang.org/stable/std/sync/atomic/struct.AtomicI8.html#method.compare_and_swap>.
macro_rules! compare_and_swap {
    ($ordering:ident, $bytes:tt, $name:ident) => {
        intrinsics! {
            #[maybe_use_optimized_c_shim]
            #[unsafe(naked)]
            pub unsafe extern "C" fn $name (
                expected: int_ty!($bytes), desired: int_ty!($bytes), ptr: *mut int_ty!($bytes)
            ) -> int_ty!($bytes) {
                // We can't use `AtomicI8::compare_and_swap`; we *are* compare_and_swap.
                core::arch::naked_asm! {
                    // UXT s(tmp0), s(0)
                    concat!(uxt!($bytes), " ", reg!($bytes, 16), ", ", reg!($bytes, 0)),
                    "0:",
                    // LDXR   s(0), [x2]
                    concat!(ldxr!($ordering, $bytes), " ", reg!($bytes, 0), ", [x2]"),
                    // cmp    s(0), s(tmp0)
                    concat!("cmp ", reg!($bytes, 0), ", ", reg!($bytes, 16)),
                    "bne    1f",
                    // STXR   w(tmp1), s(1), [x2]
                    concat!(stxr!($ordering, $bytes), " w17, ", reg!($bytes, 1), ", [x2]"),
                    "cbnz   w17, 0b",
                    "1:",
                    "ret",
                }
            }
        }
    };
}

// i128 uses a completely different impl, so it has its own macro.
macro_rules! compare_and_swap_i128 {
    ($ordering:ident, $name:ident) => {
        intrinsics! {
            #[maybe_use_optimized_c_shim]
            #[unsafe(naked)]
            pub unsafe extern "C" fn $name (
                expected: i128, desired: i128, ptr: *mut i128
            ) -> i128 {
                core::arch::naked_asm! {
                    "mov    x16, x0",
                    "mov    x17, x1",
                    "0:",
                    // LDXP   x0, x1, [x4]
                    concat!(ldxp!($ordering), " x0, x1, [x4]"),
                    "cmp    x0, x16",
                    "ccmp   x1, x17, #0, eq",
                    "bne    1f",
                    // STXP   w(tmp2), x2, x3, [x4]
                    concat!(stxp!($ordering), " w15, x2, x3, [x4]"),
                    "cbnz   w15, 0b",
                    "1:",
                    "ret",
                }
            }
        }
    };
}

/// See <https://doc.rust-lang.org/stable/std/sync/atomic/struct.AtomicI8.html#method.swap>.
macro_rules! swap {
    ($ordering:ident, $bytes:tt, $name:ident) => {
        intrinsics! {
            #[maybe_use_optimized_c_shim]
            #[unsafe(naked)]
            pub unsafe extern "C" fn $name (
                left: int_ty!($bytes), right_ptr: *mut int_ty!($bytes)
            ) -> int_ty!($bytes) {
                core::arch::naked_asm! {
                    // mov    s(tmp0), s(0)
                    concat!("mov ", reg!($bytes, 16), ", ", reg!($bytes, 0)),
                    "0:",
                    // LDXR   s(0), [x1]
                    concat!(ldxr!($ordering, $bytes), " ", reg!($bytes, 0), ", [x1]"),
                    // STXR   w(tmp1), s(tmp0), [x1]
                    concat!(stxr!($ordering, $bytes), " w17, ", reg!($bytes, 16), ", [x1]"),
                    "cbnz   w17, 0b",
                    "ret",
                }
            }
        }
    };
}

/// See (e.g.) <https://doc.rust-lang.org/stable/std/sync/atomic/struct.AtomicI8.html#method.fetch_add>.
macro_rules! fetch_op {
    ($ordering:ident, $bytes:tt, $name:ident, $op:literal) => {
        intrinsics! {
            #[maybe_use_optimized_c_shim]
            #[unsafe(naked)]
            pub unsafe extern "C" fn $name (
                val: int_ty!($bytes), ptr: *mut int_ty!($bytes)
            ) -> int_ty!($bytes) {
                core::arch::naked_asm! {
                    // mov    s(tmp0), s(0)
                    concat!("mov ", reg!($bytes, 16), ", ", reg!($bytes, 0)),
                    "0:",
                    // LDXR   s(0), [x1]
                    concat!(ldxr!($ordering, $bytes), " ", reg!($bytes, 0), ", [x1]"),
                    // OP     s(tmp1), s(0), s(tmp0)
                    concat!($op, " ", reg!($bytes, 17), ", ", reg!($bytes, 0), ", ", reg!($bytes, 16)),
                    // STXR   w(tmp2), s(tmp1), [x1]
                    concat!(stxr!($ordering, $bytes), " w15, ", reg!($bytes, 17), ", [x1]"),
                    "cbnz  w15, 0b",
                    "ret",
                }
            }
        }
    }
}

// We need a single macro to pass to `foreach_ldadd`.
macro_rules! add {
    ($ordering:ident, $bytes:tt, $name:ident) => {
        fetch_op! { $ordering, $bytes, $name, "add" }
    };
}

macro_rules! and {
    ($ordering:ident, $bytes:tt, $name:ident) => {
        fetch_op! { $ordering, $bytes, $name, "bic" }
    };
}

macro_rules! xor {
    ($ordering:ident, $bytes:tt, $name:ident) => {
        fetch_op! { $ordering, $bytes, $name, "eor" }
    };
}

macro_rules! or {
    ($ordering:ident, $bytes:tt, $name:ident) => {
        fetch_op! { $ordering, $bytes, $name, "orr" }
    };
}

#[macro_export]
macro_rules! foreach_ordering {
    ($macro:path, $bytes:tt, $name:ident) => {
        $macro!( Relaxed, $bytes, ${concat($name, _relax)} );
        $macro!( Acquire, $bytes, ${concat($name, _acq)} );
        $macro!( Release, $bytes, ${concat($name, _rel)} );
        $macro!( AcqRel, $bytes, ${concat($name, _acq_rel)} );
    };
    ($macro:path, $name:ident) => {
        $macro!( Relaxed, ${concat($name, _relax)} );
        $macro!( Acquire, ${concat($name, _acq)} );
        $macro!( Release, ${concat($name, _rel)} );
        $macro!( AcqRel, ${concat($name, _acq_rel)} );
    };
}

#[macro_export]
macro_rules! foreach_bytes {
    ($macro:path, $name:ident) => {
        foreach_ordering!( $macro, 1, ${concat(__aarch64_, $name, "1")} );
        foreach_ordering!( $macro, 2, ${concat(__aarch64_, $name, "2")} );
        foreach_ordering!( $macro, 4, ${concat(__aarch64_, $name, "4")} );
        foreach_ordering!( $macro, 8, ${concat(__aarch64_, $name, "8")} );
    };
}

/// Generate different macros for cas/swp/add/clr/eor/set so that we can test them separately.
#[macro_export]
macro_rules! foreach_cas {
    ($macro:path) => {
        foreach_bytes!($macro, cas);
    };
}

/// Only CAS supports 16 bytes, and it has a different implementation that uses a different macro.
#[macro_export]
macro_rules! foreach_cas16 {
    ($macro:path) => {
        foreach_ordering!($macro, __aarch64_cas16);
    };
}
#[macro_export]
macro_rules! foreach_swp {
    ($macro:path) => {
        foreach_bytes!($macro, swp);
    };
}
#[macro_export]
macro_rules! foreach_ldadd {
    ($macro:path) => {
        foreach_bytes!($macro, ldadd);
    };
}
#[macro_export]
macro_rules! foreach_ldclr {
    ($macro:path) => {
        foreach_bytes!($macro, ldclr);
    };
}
#[macro_export]
macro_rules! foreach_ldeor {
    ($macro:path) => {
        foreach_bytes!($macro, ldeor);
    };
}
#[macro_export]
macro_rules! foreach_ldset {
    ($macro:path) => {
        foreach_bytes!($macro, ldset);
    };
}

foreach_cas!(compare_and_swap);
foreach_cas16!(compare_and_swap_i128);
foreach_swp!(swap);
foreach_ldadd!(add);
foreach_ldclr!(and);
foreach_ldeor!(xor);
foreach_ldset!(or);
