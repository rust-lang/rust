//! Aarch64 targets have two possible implementations for atomics:
//! 1. Load-Locked, Store-Conditional (LL/SC), older and slower.
//! 2. Large System Extensions (LSE), newer and faster.
//! To avoid breaking backwards compat, C toolchains introduced a concept of "outlined atomics",
//! where atomic operations call into the compiler runtime to dispatch between two depending on
//! which is supported on the current CPU.
//! See <https://community.arm.com/arm-community-blogs/b/tools-software-ides-blog/posts/making-the-most-of-the-arm-architecture-in-gcc-10#:~:text=out%20of%20line%20atomics> for more discussion.
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

use core::sync::atomic::{AtomicU8, Ordering};

/// non-zero if the host supports LSE atomics.
static HAVE_LSE_ATOMICS: AtomicU8 = AtomicU8::new(0);

intrinsics! {
    /// Call to enable LSE in outline atomic operations. The caller must verify
    /// LSE operations are supported.
    pub extern "C" fn __rust_enable_lse() {
        HAVE_LSE_ATOMICS.store(1, Ordering::Relaxed);
    }
}

/// Function to enable/disable LSE. To be used only for testing purposes.
#[cfg(feature = "unstable-public-internals")]
pub unsafe fn set_have_lse_atomics(has_lse: bool) {
    let lse_flag = if has_lse { 1 } else { 0 };
    HAVE_LSE_ATOMICS.store(lse_flag, Ordering::Relaxed);
}

/// Function to obtain whether LSE is enabled or not. To be used only for testing purposes.
#[cfg(feature = "unstable-public-internals")]
pub fn get_have_lse_atomics() -> bool {
    HAVE_LSE_ATOMICS.load(Ordering::Relaxed) != 0
}

/// Translate a byte size to a Rust type.
#[rustfmt::skip]
macro_rules! int_ty {
    (1) => { u8 };
    (2) => { u16 };
    (4) => { u32 };
    (8) => { u64 };
    (16) => { u128 };
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
    (16, $num:literal) => { concat!("x", $num) };
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

// Translate memory ordering to the LSE suffix
#[rustfmt::skip]
macro_rules! lse_mem_sfx {
    (Relaxed) => { "" };
    (Acquire) => { "a" };
    (Release) => { "l" };
    (AcqRel) => { "al" };
}

// Generate the aarch64 LSE operation for memory ordering and width
macro_rules! lse {
    ($op:literal, $order:ident, 16) => {
        concat!($op, "p", lse_mem_sfx!($order))
    };
    ($op:literal, $order:ident, $bytes:tt) => {
        concat!($op, lse_mem_sfx!($order), size!($bytes))
    };
}

/// See <https://doc.rust-lang.org/stable/std/sync/atomic/struct.AtomicI8.html#method.compare_and_swap>.
macro_rules! compare_and_swap {
    ($ordering:ident, $bytes:tt, $name:ident) => {
        intrinsics! {
            #[maybe_use_optimized_c_shim]
            pub unsafe extern "C" fn $name (
                expected: int_ty!($bytes), desired: int_ty!($bytes), ptr: *mut int_ty!($bytes)
            ) -> int_ty!($bytes) {
                let mut expected = expected;
                unsafe {
                    if HAVE_LSE_ATOMICS.load(Ordering::Relaxed) != 0 {
                        core::arch::asm!(
                            ".arch_extension lse",
                            concat!(lse!("cas", $ordering, $bytes), " ", reg!($bytes, 0), ", ", reg!($bytes, 1),", [x2]"),
                            inlateout ( "x0" ) expected,
                            in("x1") desired,
                            in("x2") ptr,
                            options(nostack),
                        );
                    } else {
                        core::arch::asm!(
                            concat!(uxt!($bytes), " ", reg!($bytes, 16), ", ", reg!($bytes, 0)),
                            "1:",
                            concat!(ldxr!($ordering, $bytes), " ", reg!($bytes, 0), ", [x2]"),
                            concat!("cmp ", reg!($bytes, 0), ", ", reg!($bytes, 16)),
                            "bne 2f",
                            concat!(stxr!($ordering, $bytes), " w17, ", reg!($bytes, 1), ", [x2]"),
                            "cbnz w17, 1b",
                            "2:",
                            inlateout("x0") expected,
                            in("x1") desired,
                            in("x2") ptr,
                            out("x16") _,
                            out("w17") _,
                            options(nostack),
                        );
                    }
                }
                expected
            }
        }
    };
}

// u128 uses a completely different impl, so it has its own macro.
macro_rules! compare_and_swap_u128 {
    ($ordering:ident, $name:ident) => {
        intrinsics! {
            #[maybe_use_optimized_c_shim]
            pub unsafe extern "C" fn $name (
                expected: u128, desired: u128, ptr: *mut u128
            ) -> u128 {
                let mut expected_lo = (expected & 0xFFFFFFFFFFFFFFFF) as u64;
                let mut expected_hi = (expected >> 64) as u64;
                let desired_lo = (desired & 0xFFFFFFFFFFFFFFFF) as u64;
                let desired_hi = (desired >> 64) as u64;

                unsafe {
                    if HAVE_LSE_ATOMICS.load(Ordering::Relaxed) != 0 {
                        core::arch::asm!(
                            ".arch_extension lse",
                            concat!(lse!("cas", $ordering, 16), " x0, x1, x2, x3, [x4]"),
                            inlateout("x0") expected_lo,
                            inlateout("x1") expected_hi,
                            in("x2") desired_lo,
                            in("x3") desired_hi,
                            in("x4") ptr,
                            options(nostack),
                        );
                    } else {
                        core::arch::asm!(
                            "mov x16, x0",
                            "mov x17, x1",
                            "1:",
                            concat!(ldxp!($ordering), " x0, x1, [x4]"),
                            "cmp x0, x16",
                            "ccmp x1, x17, #0x0, eq",
                            "b.ne 2f",
                            concat!(stxp!($ordering), " w15, x2, x3, [x4]"),
                            "cbnz w15, 1b",
                            "2:",
                            inlateout("x0") expected_lo,
                            inlateout("x1") expected_hi,
                            in("x2") desired_lo,
                            in("x3") desired_hi,
                            in("x4") ptr,
                            out("w15") _,
                            out("x16") _,
                            out("x17") _,
                            options(nostack),
                        );
                    }
                }
                return ((expected_hi as u128) << 64) | expected_lo as u128;
            }
        }
    };
}

/// See <https://doc.rust-lang.org/stable/std/sync/atomic/struct.AtomicI8.html#method.swap>.
macro_rules! swap {
    ($ordering:ident, $bytes:tt, $name:ident) => {
        intrinsics! {
            #[maybe_use_optimized_c_shim]
            pub unsafe extern "C" fn $name (
                left: int_ty!($bytes), right_ptr: *mut int_ty!($bytes)
            ) -> int_ty!($bytes) {
                let mut left = left;
                unsafe {
                    if HAVE_LSE_ATOMICS.load(Ordering::Relaxed) != 0 {
                        core::arch::asm! {
                            ".arch_extension lse",
                            concat!( lse!("swp", $ordering, $bytes), " ", reg!($bytes, 0), ", ", reg!($bytes, 0), ", [x1]"),
                            inlateout("x0") left,
                            in("x1") right_ptr,
                            options(nostack),
                        };
                    } else {
                        core::arch::asm! {
                            concat!("mov ", reg!($bytes, 16), ", ", reg!($bytes, 0)),
                            "1:",
                            concat!(ldxr!($ordering, $bytes), " ", reg!($bytes, 0), ", [x1]"),
                            concat!(stxr!($ordering, $bytes), " w17, ", reg!($bytes, 16), ", [x1]"),
                            "cbnz w17, 1b",
                            inlateout("x0") left,
                            in("x1") right_ptr,
                            options(nostack),
                            out("x16") _,
                            out("w17") _,
                        };
                    }
                }
                left
            }
        }
    };
}

/// See (e.g.) <https://doc.rust-lang.org/stable/std/sync/atomic/struct.AtomicI8.html#method.fetch_add>.
macro_rules! fetch_op {
    ($ordering:ident, $bytes:tt, $name:ident, $op:literal, $lse_op:literal) => {
        intrinsics! {
            #[maybe_use_optimized_c_shim]
            pub unsafe extern "C" fn $name (
                val: int_ty!($bytes), ptr: *mut int_ty!($bytes)
            ) -> int_ty!($bytes) {
                unsafe {
                    if HAVE_LSE_ATOMICS.load(Ordering::Relaxed) != 0 {
                        core::arch::asm! {
                            ".arch_extension lse",
                            concat!(lse!($lse_op, $ordering, $bytes), " ", reg!($bytes, 0), ", ", reg!($bytes, 0),", [x1]"),
                            in("x0") val,
                            in("x1") ptr,
                            options(nostack),
                        };
                    } else {
                        core::arch::asm! {
                            concat!("mov ", reg!($bytes, 16), ", ", reg!($bytes, 0)),
                            "1:",
                            concat!(ldxr!($ordering, $bytes), " ", reg!($bytes, 0), ", [x1]"),
                            concat!($op, " ", reg!($bytes, 17), ", ", reg!($bytes, 0), ", ", reg!($bytes, 16)),
                            concat!(stxr!($ordering, $bytes), " w15, ", reg!($bytes, 17), ", [x1]"),
                            "cbnz w15, 1b",
                            in("x0") val,
                            in("x1") ptr,
                            out("w15") _,
                            out("x16") _,
                            out("x17") _,
                            options(nostack),
                        }
                    }
                }
                val
            }
        }
    }
}

// We need a single macro to pass to `foreach_ldadd`.
macro_rules! add {
    ($ordering:ident, $bytes:tt, $name:ident) => {
        fetch_op! { $ordering, $bytes, $name, "add", "ldadd" }
    };
}

macro_rules! and {
    ($ordering:ident, $bytes:tt, $name:ident) => {
        fetch_op! { $ordering, $bytes, $name, "bic", "ldclr" }
    };
}

macro_rules! xor {
    ($ordering:ident, $bytes:tt, $name:ident) => {
        fetch_op! { $ordering, $bytes, $name, "eor", "ldeor" }
    };
}

macro_rules! or {
    ($ordering:ident, $bytes:tt, $name:ident) => {
        fetch_op! { $ordering, $bytes, $name, "orr", "ldset" }
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
foreach_cas16!(compare_and_swap_u128);
foreach_swp!(swap);
foreach_ldadd!(add);
foreach_ldclr!(and);
foreach_ldeor!(xor);
foreach_ldset!(or);
