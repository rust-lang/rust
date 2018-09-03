// Reference: Section 7.4 "Hints" of ACLE

macro_rules! dmb_dsb {
    ($A:ident) => {
        impl super::super::sealed::Dmb for $A {
            #[inline(always)]
            unsafe fn __dmb(&self) {
                asm!(concat!("DMB ", stringify!($A)) : : : "memory" : "volatile")
            }
        }

        impl super::super::sealed::Dsb for $A {
            #[inline(always)]
            unsafe fn __dsb(&self) {
                asm!(concat!("DSB ", stringify!($A)) : : : "memory" : "volatile")
            }
        }
    };
}

mod common;

pub use self::common::*;

#[cfg(not(target_feature = "mclass"))]
mod not_mclass;

#[cfg(not(target_feature = "mclass"))]
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
