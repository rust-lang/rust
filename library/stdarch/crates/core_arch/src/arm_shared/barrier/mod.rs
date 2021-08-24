// Reference: Section 7.4 "Hints" of ACLE

// CP15 instruction
#[cfg(not(any(
    // v8
    target_arch = "aarch64",
    // v7
    target_feature = "v7",
    // v6-M
    target_feature = "mclass"
)))]
mod cp15;

#[cfg(not(any(
    target_arch = "aarch64",
    target_feature = "v7",
    target_feature = "mclass"
)))]
pub use self::cp15::*;

// Dedicated instructions
#[cfg(any(
    target_arch = "aarch64",
    target_feature = "v7",
    target_feature = "mclass"
))]
macro_rules! dmb_dsb {
    ($A:ident) => {
        impl super::super::sealed::Dmb for $A {
            #[inline(always)]
            unsafe fn __dmb(&self) {
                super::dmb(super::arg::$A)
            }
        }

        impl super::super::sealed::Dsb for $A {
            #[inline(always)]
            unsafe fn __dsb(&self) {
                super::dsb(super::arg::$A)
            }
        }
    };
}

#[cfg(any(
    target_arch = "aarch64",
    target_feature = "v7",
    target_feature = "mclass"
))]
mod common;

#[cfg(any(
    target_arch = "aarch64",
    target_feature = "v7",
    target_feature = "mclass"
))]
pub use self::common::*;

#[cfg(any(target_arch = "aarch64", target_feature = "v7",))]
mod not_mclass;

#[cfg(any(target_arch = "aarch64", target_feature = "v7",))]
pub use self::not_mclass::*;

#[cfg(target_arch = "aarch64")]
mod v8;

#[cfg(target_arch = "aarch64")]
pub use self::v8::*;

/// Generates a DMB (data memory barrier) instruction or equivalent CP15 instruction.
///
/// DMB ensures the observed ordering of memory accesses. Memory accesses of the specified type
/// issued before the DMB are guaranteed to be observed (in the specified scope) before memory
/// accesses issued after the DMB.
///
/// For example, DMB should be used between storing data, and updating a flag variable that makes
/// that data available to another core.
///
/// The __dmb() intrinsic also acts as a compiler memory barrier of the appropriate type.
#[inline(always)]
pub unsafe fn __dmb<A>(arg: A)
where
    A: super::sealed::Dmb,
{
    arg.__dmb()
}

/// Generates a DSB (data synchronization barrier) instruction or equivalent CP15 instruction.
///
/// DSB ensures the completion of memory accesses. A DSB behaves as the equivalent DMB and has
/// additional properties. After a DSB instruction completes, all memory accesses of the specified
/// type issued before the DSB are guaranteed to have completed.
///
/// The __dsb() intrinsic also acts as a compiler memory barrier of the appropriate type.
#[inline(always)]
pub unsafe fn __dsb<A>(arg: A)
where
    A: super::sealed::Dsb,
{
    arg.__dsb()
}

/// Generates an ISB (instruction synchronization barrier) instruction or equivalent CP15
/// instruction.
///
/// This instruction flushes the processor pipeline fetch buffers, so that following instructions
/// are fetched from cache or memory.
///
/// An ISB is needed after some system maintenance operations. An ISB is also needed before
/// transferring control to code that has been loaded or modified in memory, for example by an
/// overlay mechanism or just-in-time code generator.  (Note that if instruction and data caches are
/// separate, privileged cache maintenance operations would be needed in order to unify the caches.)
///
/// The only supported argument for the __isb() intrinsic is 15, corresponding to the SY (full
/// system) scope of the ISB instruction.
#[inline(always)]
pub unsafe fn __isb<A>(arg: A)
where
    A: super::sealed::Isb,
{
    arg.__isb()
}

extern "unadjusted" {
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.dmb")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.dmb")]
    fn dmb(_: i32);

    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.dsb")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.dsb")]
    fn dsb(_: i32);

    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.isb")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.isb")]
    fn isb(_: i32);
}

// we put these in a module to prevent weirdness with glob re-exports
mod arg {
    // See Section 7.3  Memory barriers of ACLE
    pub const SY: i32 = 15;
    pub const ST: i32 = 14;
    pub const LD: i32 = 13;
    pub const ISH: i32 = 11;
    pub const ISHST: i32 = 10;
    pub const ISHLD: i32 = 9;
    pub const NSH: i32 = 7;
    pub const NSHST: i32 = 6;
    pub const NSHLD: i32 = 5;
    pub const OSH: i32 = 3;
    pub const OSHST: i32 = 2;
    pub const OSHLD: i32 = 1;
}
