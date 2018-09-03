//! ARM C Language Extensions (ACLE)
//!
//! # Developer notes
//!
//! Below is a list of built-in targets that are representative of the different ARM
//! architectures; the list includes the `target_feature`s they possess.
//!
//! - `armv4t-unknown-linux-gnueabi` - **ARMv4** - `+v4t`
//! - `armv5te-unknown-linux-gnueabi` - **ARMv5TE** - `+v4t +v5te`
//! - `arm-unknown-linux-gnueabi` - **ARMv6** - `+v4t +v5te +v6`
//! - `thumbv6m-none-eabi` - **ARMv6-M** - `+v4t +v5te +v6 +thumb-mode +mclass`
//! - `armv7-unknown-linux-gnueabihf` - **ARMv7-A** - `+v4t +v5te +v6 +v6k +v6t2 +v7 +dsp +thumb2 +aclass`
//! - `armv7r-none-eabi` - **ARMv7-R** - `+v4t +v5te +v6 +v6k +v6t2  +v7 +dsp +thumb2 +rclass`
//! - `thumbv7m-none-eabi` - **ARMv7-M** - `+v4t +v5te +v6 +v6k +v6t2 +v7 +thumb2 +thumb-mode +mclass`
//! - `thumbv7em-none-eabi` - **ARMv7E-M** - `+v4t +v5te +v6 +v6k +v6t2 +v7 +dsp +thumb2 +thumb-mode +mclass`
//! - `aarch64-unknown-linux-gnu` - **ARMv8-A (AArch64)** - `+fp +neon`
//!
//! Section 10.1 of ACLE says:
//!
//! - "In the sequence of Arm architectures { v5, v5TE, v6, v6T2, v7 } each architecture includes
//! its predecessor instruction set."
//!
//! - "In the sequence of Thumb-only architectures { v6-M, v7-M, v7E-M } each architecture includes
//! its predecessor instruction set."
//!
//! From that info and from looking at how LLVM features work (using custom targets) we can identify
//! features that are subsets of others:
//!
//! Legend: `a < b` reads as "`a` is a subset of `b`"; this means that if `b` is enabled then `a` is
//! enabled as well.
//!
//! - `v4t < v5te < v6 < v6k < v6t2 < v7 < v8`
//! - `v6 < v8m < v6t2`
//! - `v7 < v8m.main`
//!
//! # References
//!
//! - [ACLE Q2 2018](https://developer.arm.com/docs/101028/latest)

// Supported arches: 8, 7, 6-M. See Section 10.1 of ACLE (e.g. DMB)
// But this is further refined within the module
#[cfg(any(
    // v8
    target_arch = "aarch64",
    // v7
    target_feature = "v7",
    // v6-M
    target_feature = "mclass"
))]
mod barrier;

#[cfg(any(
    target_arch = "aarch64",
    target_feature = "v7",
    target_feature = "mclass"
))]
pub use self::barrier::*;

mod hints;

pub use self::hints::*;

mod registers;

pub use self::registers::*;

// Supported arches: 5TE, 7E-M. See Section 10.1 of ACLE (e.g. QADD)
// But we also exclude the A profile because DSP is deprecated on that profile as of ACLE 2.0 (see
// section 5.4.7)
#[cfg(any(
    // >= v5TE but excludes v7-A
    all(target_feature = "v5te", not(target_feature = "mclass"), not(target_feature = "aclass")),
    // v7E-M
    all(target_feature = "mclass", target_feature = "dsp"),
))]
mod dsp;

#[cfg(any(
    all(
        target_feature = "v5te",
        not(target_feature = "mclass"),
        not(target_feature = "aclass")
    ),
    all(target_feature = "mclass", target_feature = "dsp"),
))]
pub use dsp::*;

// Deprecated in ACLE 2.0 for the A profile but fully supported on the M and R profiles, says
// Section 5.4.9 of ACLE.
#[cfg(any(
    // v7-R
    target_feature = "rclass",
    // v7E-M
    all(target_feature = "mclass", target_feature = "dsp")
))]
mod simd32;

mod sealed {
    pub trait Dmb {
        unsafe fn __dmb(&self);
    }

    pub trait Dsb {
        unsafe fn __dsb(&self);
    }

    pub trait Isb {
        unsafe fn __isb(&self);
    }

    pub trait Rsr {
        unsafe fn __rsr(&self) -> u32;
    }

    pub trait Rsr64 {
        unsafe fn __rsr64(&self) -> u64;
    }

    pub trait Rsrp {
        unsafe fn __rsrp(&self) -> *const u8;
    }

    pub trait Wsr {
        unsafe fn __wsr(&self, value: u32);
    }

    pub trait Wsr64 {
        unsafe fn __wsr64(&self, value: u64);
    }

    pub trait Wsrp {
        unsafe fn __wsrp(&self, value: *const u8);
    }
}
