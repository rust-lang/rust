// TODO: when `unsafe_block_in_unsafe_fn` is stabilized, remove this
#![allow(unused_unsafe)]
// The functions are complex with many branches, and explicit
// `return`s makes it clear where function exit points are
#![allow(clippy::needless_return)]
#![allow(clippy::comparison_chain)]
// Clippy is confused by the complex configuration
#![allow(clippy::if_same_then_else)]
#![allow(clippy::needless_bool)]

//! This `specialized_div_rem` module is originally from version 1.0.0 of the
//! `specialized-div-rem` crate. Note that `for` loops with ranges are not used in this
//! module, since unoptimized compilation may generate references to `memcpy`.
//!
//! The purpose of these macros is to easily change the both the division algorithm used
//! for a given integer size and the half division used by that algorithm. The way
//! functions call each other is also constructed such that linkers will find the chain of
//! software and hardware divisions needed for every size of signed and unsigned division.
//! For example, most target compilations do the following:
//!
//!  - Many 128 bit division functions like `u128::wrapping_div` use
//!    `std::intrinsics::unchecked_div`, which gets replaced by `__udivti3` because there
//!    is not a 128 bit by 128 bit hardware division function in most architectures.
//!    `__udivti3` uses `u128_div_rem` (this extra level of function calls exists because
//!    `__umodti3` and `__udivmodti4` also exist, and `specialized_div_rem` supplies just
//!    one function to calculate both the quotient and remainder. If configuration flags
//!    enable it, `impl_trifecta!` defines `u128_div_rem` to use the trifecta algorithm,
//!    which requires the half sized division `u64_by_u64_div_rem`. If the architecture
//!    supplies a 64 bit hardware division instruction, `u64_by_u64_div_rem` will be
//!    reduced to those instructions. Note that we do not specify the half size division
//!    directly to be `__udivdi3`, because hardware division would never be introduced.
//!  - If the architecture does not supply a 64 bit hardware division instruction, u64
//!    divisions will use functions such as `__udivdi3`. This will call `u64_div_rem`
//!    which is defined by `impl_delegate!`. The half division for this algorithm is
//!    `u32_by_u32_div_rem` which in turn becomes hardware division instructions or more
//!    software division algorithms.
//!  - If the architecture does not supply a 32 bit hardware instruction, linkers will
//!    look for `__udivsi3`. `impl_binary_long!` is used, but this  algorithm uses no half
//!    division, so the chain of calls ends here.
//!
//! On some architectures like x86_64, an asymmetrically sized division is supplied, in
//! which 128 bit numbers can be divided by 64 bit numbers. `impl_asymmetric!` is used to
//! extend the 128 by 64 bit division to a full 128 by 128 bit division.

// `allow(dead_code)` is used in various places, because the configuration code would otherwise be
// ridiculously complex

#[macro_use]
mod norm_shift;

#[macro_use]
mod binary_long;

#[macro_use]
mod delegate;

// used on SPARC
#[allow(unused_imports)]
#[cfg(not(feature = "public-test-deps"))]
pub(crate) use self::delegate::u128_divide_sparc;

#[cfg(feature = "public-test-deps")]
pub use self::delegate::u128_divide_sparc;

#[macro_use]
mod trifecta;

#[macro_use]
mod asymmetric;

/// The behavior of all divisions by zero is controlled by this function. This function should be
/// impossible to reach by Rust users, unless `compiler-builtins` public division functions or
/// `core/std::unchecked_div/rem` are directly used without a zero check in front.
fn zero_div_fn() -> ! {
    unsafe { core::hint::unreachable_unchecked() }
}

const USE_LZ: bool = {
    if cfg!(target_arch = "arm") {
        if cfg!(target_feature = "thumb-mode") {
            // ARM thumb targets have CLZ instructions if the instruction set of ARMv6T2 is
            // supported. This is needed to successfully differentiate between targets like
            // `thumbv8.base` and `thumbv8.main`.
            cfg!(target_feature = "v6t2")
        } else {
            // Regular ARM targets have CLZ instructions if the ARMv5TE instruction set is
            // supported. Technically, ARMv5T was the first to have CLZ, but the "v5t" target
            // feature does not seem to work.
            cfg!(target_feature = "v5te")
        }
    } else if cfg!(any(target_arch = "sparc", target_arch = "sparc64")) {
        // LZD or LZCNT on SPARC only exists for the VIS 3 extension and later.
        cfg!(target_feature = "vis3")
    } else if cfg!(any(target_arch = "riscv32", target_arch = "riscv64")) {
        // The `B` extension on RISC-V determines if a CLZ assembly instruction exists
        cfg!(target_feature = "b")
    } else {
        // All other common targets Rust supports should have CLZ instructions
        true
    }
};

impl_normalization_shift!(
    u32_normalization_shift,
    USE_LZ,
    32,
    u32,
    i32,
    allow(dead_code)
);
impl_normalization_shift!(
    u64_normalization_shift,
    USE_LZ,
    64,
    u64,
    i64,
    allow(dead_code)
);

/// Divides `duo` by `div` and returns a tuple of the quotient and the remainder.
/// `checked_div` and `checked_rem` are used to avoid bringing in panic function
/// dependencies.
#[inline]
fn u64_by_u64_div_rem(duo: u64, div: u64) -> (u64, u64) {
    if let Some(quo) = duo.checked_div(div) {
        if let Some(rem) = duo.checked_rem(div) {
            return (quo, rem);
        }
    }
    zero_div_fn()
}

// Whether `trifecta` or `delegate` is faster for 128 bit division depends on the speed at which a
// microarchitecture can multiply and divide. We decide to be optimistic and assume `trifecta` is
// faster if the target pointer width is at least 64.
#[cfg(all(
    not(any(target_pointer_width = "16", target_pointer_width = "32")),
    not(all(not(feature = "no-asm"), target_arch = "x86_64")),
    not(any(target_arch = "sparc", target_arch = "sparc64"))
))]
impl_trifecta!(
    u128_div_rem,
    zero_div_fn,
    u64_by_u64_div_rem,
    32,
    u32,
    u64,
    u128
);

// If the pointer width less than 64, then the target architecture almost certainly does not have
// the fast 64 to 128 bit widening multiplication needed for `trifecta` to be faster.
#[cfg(all(
    any(target_pointer_width = "16", target_pointer_width = "32"),
    not(all(not(feature = "no-asm"), target_arch = "x86_64")),
    not(any(target_arch = "sparc", target_arch = "sparc64"))
))]
impl_delegate!(
    u128_div_rem,
    zero_div_fn,
    u64_normalization_shift,
    u64_by_u64_div_rem,
    32,
    u32,
    u64,
    u128,
    i128
);

/// Divides `duo` by `div` and returns a tuple of the quotient and the remainder.
///
/// # Safety
///
/// If the quotient does not fit in a `u64`, a floating point exception occurs.
/// If `div == 0`, then a division by zero exception occurs.
#[cfg(all(not(feature = "no-asm"), target_arch = "x86_64"))]
#[inline]
unsafe fn u128_by_u64_div_rem(duo: u128, div: u64) -> (u64, u64) {
    let duo_lo = duo as u64;
    let duo_hi = (duo >> 64) as u64;
    let quo: u64;
    let rem: u64;
    unsafe {
        // divides the combined registers rdx:rax (`duo` is split into two 64 bit parts to do this)
        // by `div`. The quotient is stored in rax and the remainder in rdx.
        // FIXME: Use the Intel syntax once we drop LLVM 9 support on rust-lang/rust.
        core::arch::asm!(
            "div {0}",
            in(reg) div,
            inlateout("rax") duo_lo => quo,
            inlateout("rdx") duo_hi => rem,
            options(att_syntax, pure, nomem, nostack)
        );
    }
    (quo, rem)
}

// use `asymmetric` instead of `trifecta` on x86_64
#[cfg(all(not(feature = "no-asm"), target_arch = "x86_64"))]
impl_asymmetric!(
    u128_div_rem,
    zero_div_fn,
    u64_by_u64_div_rem,
    u128_by_u64_div_rem,
    32,
    u32,
    u64,
    u128
);

/// Divides `duo` by `div` and returns a tuple of the quotient and the remainder.
/// `checked_div` and `checked_rem` are used to avoid bringing in panic function
/// dependencies.
#[inline]
#[allow(dead_code)]
fn u32_by_u32_div_rem(duo: u32, div: u32) -> (u32, u32) {
    if let Some(quo) = duo.checked_div(div) {
        if let Some(rem) = duo.checked_rem(div) {
            return (quo, rem);
        }
    }
    zero_div_fn()
}

// When not on x86 and the pointer width is not 64, use `delegate` since the division size is larger
// than register size.
#[cfg(all(
    not(all(not(feature = "no-asm"), target_arch = "x86")),
    not(target_pointer_width = "64")
))]
impl_delegate!(
    u64_div_rem,
    zero_div_fn,
    u32_normalization_shift,
    u32_by_u32_div_rem,
    16,
    u16,
    u32,
    u64,
    i64
);

// When not on x86 and the pointer width is 64, use `binary_long`.
#[cfg(all(
    not(all(not(feature = "no-asm"), target_arch = "x86")),
    target_pointer_width = "64"
))]
impl_binary_long!(
    u64_div_rem,
    zero_div_fn,
    u64_normalization_shift,
    64,
    u64,
    i64
);

/// Divides `duo` by `div` and returns a tuple of the quotient and the remainder.
///
/// # Safety
///
/// If the quotient does not fit in a `u32`, a floating point exception occurs.
/// If `div == 0`, then a division by zero exception occurs.
#[cfg(all(not(feature = "no-asm"), target_arch = "x86"))]
#[inline]
unsafe fn u64_by_u32_div_rem(duo: u64, div: u32) -> (u32, u32) {
    let duo_lo = duo as u32;
    let duo_hi = (duo >> 32) as u32;
    let quo: u32;
    let rem: u32;
    unsafe {
        // divides the combined registers rdx:rax (`duo` is split into two 32 bit parts to do this)
        // by `div`. The quotient is stored in rax and the remainder in rdx.
        // FIXME: Use the Intel syntax once we drop LLVM 9 support on rust-lang/rust.
        core::arch::asm!(
            "div {0}",
            in(reg) div,
            inlateout("rax") duo_lo => quo,
            inlateout("rdx") duo_hi => rem,
            options(att_syntax, pure, nomem, nostack)
        );
    }
    (quo, rem)
}

// use `asymmetric` instead of `delegate` on x86
#[cfg(all(not(feature = "no-asm"), target_arch = "x86"))]
impl_asymmetric!(
    u64_div_rem,
    zero_div_fn,
    u32_by_u32_div_rem,
    u64_by_u32_div_rem,
    16,
    u16,
    u32,
    u64
);

// 32 bits is the smallest division used by `compiler-builtins`, so we end with binary long division
impl_binary_long!(
    u32_div_rem,
    zero_div_fn,
    u32_normalization_shift,
    32,
    u32,
    i32
);
