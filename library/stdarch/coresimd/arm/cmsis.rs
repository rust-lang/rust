#![allow(non_snake_case)]

/// Extracted from [CMSIS 5]'s CMSIS/Core/Include/cmsis_gcc.h
///
/// [CMSIS 5]: https://github.com/ARM-software/CMSIS_5

/* Core function access */

/// Enable IRQ Interrupts
///
/// Enables IRQ interrupts by clearing the I-bit in the CPSR. Can only be
/// executed in Privileged modes.
#[inline(always)]
pub unsafe fn __enable_irq() {
    asm!("cpsie i" : : : "memory" : "volatile");
}

/// Disable IRQ Interrupts
///
/// Disables IRQ interrupts by setting the I-bit in the CPSR. Can only be
/// executed in Privileged modes.
#[inline(always)]
pub unsafe fn __disable_irq() {
    asm!("cpsid i" : : : "memory" : "volatile");
}

/// Get Control Register
///
/// Returns the content of the Control Register.
#[inline(always)]
pub unsafe fn __get_CONTROL() -> u32 {
    let result: u32;
    asm!("mrs $0, CONTROL" : "=r"(result) : : : "volatile");
    result
}

/// Set Control Register
///
/// Writes the given value to the Control Register.
#[inline(always)]
pub unsafe fn __set_CONTROL(control: u32) {
    asm!("msr CONTROL, $0" : : "r"(control) : "memory" : "volatile");
}

/// Get IPSR Register
///
/// Returns the content of the IPSR Register.
#[inline(always)]
pub unsafe fn __get_IPSR() -> u32 {
    let result: u32;
    asm!("mrs $0, IPSR" : "=r"(result) : : : "volatile");
    result
}

/// Get APSR Register
///
/// Returns the content of the APSR Register.
#[inline(always)]
pub unsafe fn __get_APSR() -> u32 {
    let result: u32;
    asm!("mrs $0, APSR" : "=r"(result) : : : "volatile");
    result
}

/// Get xPSR Register
///
/// Returns the content of the xPSR Register.
#[inline(always)]
pub unsafe fn __get_xPSR() -> u32 {
    let result: u32;
    asm!("mrs $0, XPSR" : "=r"(result) : : : "volatile");
    result
}

/// Get Process Stack Pointer
///
/// Returns the current value of the Process Stack Pointer (PSP).
#[inline(always)]
pub unsafe fn __get_PSP() -> u32 {
    let result: u32;
    asm!("mrs $0, PSP" : "=r"(result) : : : "volatile");
    result
}

/// Set Process Stack Pointer
///
/// Assigns the given value to the Process Stack Pointer (PSP).
#[inline(always)]
pub unsafe fn __set_PSP(top_of_proc_stack: u32) {
    asm!("msr PSP, $0" : : "r"(top_of_proc_stack) : : "volatile");
}

/// Get Main Stack Pointer
///
/// Returns the current value of the Main Stack Pointer (MSP).
#[inline(always)]
pub unsafe fn __get_MSP() -> u32 {
    let result: u32;
    asm!("mrs $0, MSP" : "=r"(result) : : : "volatile");
    result
}

/// Set Main Stack Pointer
///
/// Assigns the given value to the Main Stack Pointer (MSP).
#[inline(always)]
pub unsafe fn __set_MSP(top_of_main_stack: u32) {
    asm!("msr MSP, $0" : : "r"(top_of_main_stack) : : "volatile");
}

/// Get Priority Mask
///
/// Returns the current state of the priority mask bit from the Priority Mask
/// Register.
#[inline(always)]
pub unsafe fn __get_PRIMASK() -> u32 {
    let result: u32;
    asm!("mrs $0, PRIMASK" : "=r"(result) : : "memory" : "volatile");
    result
}

/// Set Priority Mask
///
/// Assigns the given value to the Priority Mask Register.
#[inline(always)]
pub unsafe fn __set_PRIMASK(pri_mask: u32) {
    asm!("msr PRIMASK, $0" : : "r"(pri_mask) : : "volatile");
}

#[cfg(target_feature = "v7")]
mod v7 {
    /// Enable FIQ
    ///
    /// Enables FIQ interrupts by clearing the F-bit in the CPSR. Can only be
    /// executed in Privileged modes.
    #[inline(always)]
    pub unsafe fn __enable_fault_irq() {
        asm!("cpsie f" : : : "memory" : "volatile");
    }

    /// Disable FIQ
    ///
    /// Disables FIQ interrupts by setting the F-bit in the CPSR. Can only be
    /// executed in Privileged modes.
    #[inline(always)]
    pub unsafe fn __disable_fault_irq() {
        asm!("cpsid f" : : : "memory" : "volatile");
    }

    /// Get Base Priority
    ///
    /// Returns the current value of the Base Priority register.
    #[inline(always)]
    pub unsafe fn __get_BASEPRI() -> u32 {
        let result: u32;
        asm!("mrs $0, BASEPRI" : "=r"(result) : : : "volatile");
        result
    }

    /// Set Base Priority
    ///
    /// Assigns the given value to the Base Priority register.
    #[inline(always)]
    pub unsafe fn __set_BASEPRI(base_pri: u32) {
        asm!("msr BASEPRI, $0" : : "r"(base_pri) : "memory" : "volatile");
    }

    /// Set Base Priority with condition
    ///
    /// Assigns the given value to the Base Priority register only if BASEPRI
    /// masking is disabled, or the new value increases the BASEPRI
    /// priority level.
    #[inline(always)]
    pub unsafe fn __set_BASEPRI_MAX(base_pri: u32) {
        asm!("msr BASEPRI_MAX, $0" : : "r"(base_pri) : "memory" : "volatile");
    }

    /// Get Fault Mask
    ///
    /// Returns the current value of the Fault Mask register.
    #[inline(always)]
    pub unsafe fn __get_FAULTMASK() -> u32 {
        let result: u32;
        asm!("mrs $0, FAULTMASK" : "=r"(result) : : : "volatile");
        result
    }

    /// Set Fault Mask
    ///
    /// Assigns the given value to the Fault Mask register.
    #[inline(always)]
    pub unsafe fn __set_FAULTMASK(fault_mask: u32) {
        asm!("msr FAULTMASK, $0" : : "r"(fault_mask) : "memory" : "volatile");
    }
}

#[cfg(target_feature = "v7")]
pub use self::v7::*;

/* Core instruction access */

/// No Operation
///
/// No Operation does nothing. This instruction can be used for code alignment
/// purposes.
#[inline(always)]
pub unsafe fn __NOP() {
    asm!("nop" : : : : "volatile");
}

/// Wait For Interrupt
///
/// Wait For Interrupt is a hint instruction that suspends execution until one
/// of a number of events occurs.
#[inline(always)]
pub unsafe fn __WFI() {
    asm!("wfi" : : : : "volatile");
}

/// Wait For Event
///
/// Wait For Event is a hint instruction that permits the processor to enter a
/// low-power state until one of a number of events occurs.
#[inline(always)]
pub unsafe fn __WFE() {
    asm!("wfe" : : : : "volatile");
}

/// Send Event
///
/// Send Event is a hint instruction. It causes an event to be signaled to the
/// CPU.
#[inline(always)]
pub unsafe fn __SEV() {
    asm!("sev" : : : : "volatile");
}

/// Instruction Synchronization Barrier
///
/// Instruction Synchronization Barrier flushes the pipeline in the processor,
/// so that all instructions following the ISB are fetched from cache or
/// memory, after the instruction has been completed.
#[inline(always)]
pub unsafe fn __ISB() {
    asm!("isb 0xF" : : : "memory" : "volatile");
}

/// Data Synchronization Barrier
///
/// Acts as a special kind of Data Memory Barrier. It completes when all
/// explicit memory accesses before this instruction complete.
#[inline(always)]
pub unsafe fn __DSB() {
    asm!("dsb 0xF" : : : "memory" : "volatile");
}

/// Data Memory Barrier
///
/// Ensures the apparent order of the explicit memory operations before and
/// after the instruction, without ensuring their completion.
#[inline(always)]
pub unsafe fn __DMB() {
    asm!("dmb 0xF" : : : "memory" : "volatile");
}
