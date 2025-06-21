// # References
//
// - Section 7.4 "Hints" of ACLE
// - Section 7.7 "NOP" of ACLE

/// Generates a WFI (wait for interrupt) hint instruction, or nothing.
///
/// The WFI instruction allows (but does not require) the processor to enter a
/// low-power state until one of a number of asynchronous events occurs.
// Section 10.1 of ACLE says that the supported arches are: 8, 6K, 6-M
// LLVM says "instruction requires: armv6k"
#[cfg(any(
    target_feature = "v6",
    target_arch = "aarch64",
    target_arch = "arm64ec",
    doc
))]
#[inline(always)]
#[unstable(feature = "stdarch_arm_hints", issue = "117218")]
pub unsafe fn __wfi() {
    hint(HINT_WFI);
}

/// Generates a WFE (wait for event) hint instruction, or nothing.
///
/// The WFE instruction allows (but does not require) the processor to enter a
/// low-power state until some event occurs such as a SEV being issued by
/// another processor.
// Section 10.1 of ACLE says that the supported arches are: 8, 6K, 6-M
// LLVM says "instruction requires: armv6k"
#[cfg(any(
    target_feature = "v6",
    target_arch = "aarch64",
    target_arch = "arm64ec",
    doc
))]
#[inline(always)]
#[unstable(feature = "stdarch_arm_hints", issue = "117218")]
pub unsafe fn __wfe() {
    hint(HINT_WFE);
}

/// Generates a SEV (send a global event) hint instruction.
///
/// This causes an event to be signaled to all processors in a multiprocessor
/// system. It is a NOP on a uniprocessor system.
// Section 10.1 of ACLE says that the supported arches are: 8, 6K, 6-M, 7-M
// LLVM says "instruction requires: armv6k"
#[cfg(any(
    target_feature = "v6",
    target_arch = "aarch64",
    target_arch = "arm64ec",
    doc
))]
#[inline(always)]
#[unstable(feature = "stdarch_arm_hints", issue = "117218")]
pub unsafe fn __sev() {
    hint(HINT_SEV);
}

/// Generates a send a local event hint instruction.
///
/// This causes an event to be signaled to only the processor executing this
/// instruction. In a multiprocessor system, it is not required to affect the
/// other processors.
// LLVM says "instruction requires: armv8"
#[cfg(any(
    target_feature = "v8", // 32-bit ARMv8
    target_arch = "aarch64", // AArch64
    target_arch = "arm64ec", // Arm64EC
    doc,
))]
#[inline(always)]
#[unstable(feature = "stdarch_arm_hints", issue = "117218")]
pub unsafe fn __sevl() {
    hint(HINT_SEVL);
}

/// Generates a YIELD hint instruction.
///
/// This enables multithreading software to indicate to the hardware that it is
/// performing a task, for example a spin-lock, that could be swapped out to
/// improve overall system performance.
// Section 10.1 of ACLE says that the supported arches are: 8, 6K, 6-M
// LLVM says "instruction requires: armv6k"
#[cfg(any(
    target_feature = "v6",
    target_arch = "aarch64",
    target_arch = "arm64ec",
    doc
))]
#[inline(always)]
#[unstable(feature = "stdarch_arm_hints", issue = "117218")]
pub unsafe fn __yield() {
    hint(HINT_YIELD);
}

/// Generates an unspecified no-op instruction.
///
/// Note that not all architectures provide a distinguished NOP instruction. On
/// those that do, it is unspecified whether this intrinsic generates it or
/// another instruction. It is not guaranteed that inserting this instruction
/// will increase execution time.
#[inline(always)]
#[unstable(feature = "stdarch_arm_hints", issue = "117218")]
pub unsafe fn __nop() {
    crate::arch::asm!("nop", options(nomem, nostack, preserves_flags));
}

unsafe extern "unadjusted" {
    #[cfg_attr(
        any(target_arch = "aarch64", target_arch = "arm64ec"),
        link_name = "llvm.aarch64.hint"
    )]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.hint")]
    fn hint(_: i32);
}

// from LLVM 7.0.1's lib/Target/ARM/{ARMInstrThumb,ARMInstrInfo,ARMInstrThumb2}.td
const HINT_NOP: i32 = 0;
const HINT_YIELD: i32 = 1;
const HINT_WFE: i32 = 2;
const HINT_WFI: i32 = 3;
const HINT_SEV: i32 = 4;
const HINT_SEVL: i32 = 5;
