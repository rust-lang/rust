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
#[cfg(any(target_feature = "v6k", target_arch = "aarch64"))]
#[inline(always)]
pub unsafe fn __wfi() {
    asm!("WFI" : : : : "volatile")
}

/// Generates a WFE (wait for event) hint instruction, or nothing.
///
/// The WFE instruction allows (but does not require) the processor to enter a
/// low-power state until some event occurs such as a SEV being issued by
/// another processor.
// Section 10.1 of ACLE says that the supported arches are: 8, 6K, 6-M
// LLVM says "instruction requires: armv6k"
#[cfg(any(target_feature = "v6k", target_arch = "aarch64"))]
#[inline(always)]
pub unsafe fn __wfe() {
    asm!("WFE" : : : : "volatile")
}

/// Generates a SEV (send a global event) hint instruction.
///
/// This causes an event to be signaled to all processors in a multiprocessor
/// system. It is a NOP on a uniprocessor system.
// Section 10.1 of ACLE says that the supported arches are: 8, 6K, 6-M, 7-M
// LLVM says "instruction requires: armv6k"
#[cfg(any(target_feature = "v6k", target_arch = "aarch64"))]
#[inline(always)]
pub unsafe fn __sev() {
    asm!("SEV" : : : : "volatile")
}

/// Generates a send a local event hint instruction.
///
/// This causes an event to be signaled to only the processor executing this
/// instruction. In a multiprocessor system, it is not required to affect the
/// other processors.
// LLVM says "instruction requires: armv8"
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn __sevl() {
    asm!("SEVL" : : : : "volatile")
}

/// Generates a YIELD hint instruction.
///
/// This enables multithreading software to indicate to the hardware that it is
/// performing a task, for example a spin-lock, that could be swapped out to
/// improve overall system performance.
// Section 10.1 of ACLE says that the supported arches are: 8, 6K, 6-M
// LLVM says "instruction requires: armv6k"
#[cfg(any(target_feature = "v6k", target_arch = "aarch64"))]
#[inline(always)]
pub unsafe fn __yield() {
    asm!("YIELD" : : : : "volatile")
}

/// Generates a DBG instruction.
///
/// This provides a hint to debugging and related systems. The argument must be
/// a constant integer from 0 to 15 inclusive. See implementation documentation
/// for the effect (if any) of this instruction and the meaning of the
/// argument. This is available only when compliling for AArch32.
// Section 10.1 of ACLE says that the supported arches are: 7, 7-M
// LLVM says "instruction requires: thumb2" OR "instruction requires: armv7"
#[cfg(target_feature = "v6t2")]
#[inline(always)]
#[rustc_args_required_const(0)]
pub unsafe fn __dbg(imm4: u32) {
    macro_rules! call {
        ($imm4:expr) => {
            asm!(concat!("DBG ", stringify!($imm4)) : : : : "volatile")
        }
    }

    match imm4 & 0b1111 {
        0 => call!(0),
        1 => call!(1),
        2 => call!(2),
        3 => call!(3),
        4 => call!(4),
        5 => call!(5),
        6 => call!(6),
        7 => call!(7),
        8 => call!(8),
        9 => call!(9),
        10 => call!(10),
        11 => call!(11),
        12 => call!(12),
        13 => call!(13),
        14 => call!(14),
        _ => call!(15),
    }
}

/// Generates an unspecified no-op instruction.
///
/// Note that not all architectures provide a distinguished NOP instruction. On
/// those that do, it is unspecified whether this intrinsic generates it or
/// another instruction. It is not guaranteed that inserting this instruction
/// will increase execution time.
#[inline(always)]
pub unsafe fn __nop() {
    asm!("NOP" : : : : "volatile")
}
