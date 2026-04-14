use super::paging::AArch64AddressSpace;
#[allow(unused_imports)]
use core::arch::{asm, naked_asm};
use kernel::UserTaskSpec;

#[derive(Copy, Clone, Default)]
pub struct AArch64Context(pub [u64; 13]); // x19-x29, lr, sp

/// Context switch between two tasks.
/// Saves callee-saved registers to `old` and restores from `new`.
#[unsafe(naked)]
pub unsafe extern "C" fn context_switch(_old: *mut u64, _new: *const u64) {
    naked_asm!(
        // Save old context (x0 = old)
        "stp x19, x20, [x0, #0]",
        "stp x21, x22, [x0, #16]",
        "stp x23, x24, [x0, #32]",
        "stp x25, x26, [x0, #48]",
        "stp x27, x28, [x0, #64]",
        "stp x29, x30, [x0, #80]", // x30 = lr
        "mov x9, sp",
        "str x9, [x0, #96]",
        // Load new context (x1 = new)
        "ldp x19, x20, [x1, #0]",
        "ldp x21, x22, [x1, #16]",
        "ldp x23, x24, [x1, #32]",
        "ldp x25, x26, [x1, #48]",
        "ldp x27, x28, [x1, #64]",
        "ldp x29, x30, [x1, #80]",
        "ldr x9, [x1, #96]",
        "mov sp, x9",
        "ret",
    );
}

pub fn init_kernel_context(
    entry: extern "C" fn(usize) -> !,
    stack_top: u64,
    arg: usize,
) -> AArch64Context {
    let mut ctx = AArch64Context::default();

    // When context_switch loads this context:
    // - lr (x30) should point to our trampoline
    // - sp should be the new stack
    // - We store entry in x19 and arg in x20 for the trampoline to use
    ctx.0[11] = trampoline as *const () as u64; // lr (position 11 = x30)
    ctx.0[12] = stack_top; // sp (position 12)
    ctx.0[0] = entry as *const () as u64; // x19 = entry
    ctx.0[1] = arg as u64; // x20 = arg

    ctx
}

pub fn init_user_context(
    spec: UserTaskSpec<AArch64AddressSpace>,
    kstack_top: u64,
) -> AArch64Context {
    let mut ctx = AArch64Context::default();
    // When context_switch restores this context, lr (x30) points to
    // user_trampoline which performs the EL1 → EL0 transition.
    // Callee-saved registers carry the user task parameters:
    //   x19 = user entry_pc
    //   x20 = user stack_top
    //   x21 = TTBR0 value (address space)
    //   x22 = startup argument (forwarded to x0 in user mode)
    ctx.0[11] = user_trampoline as *const () as u64; // lr = user_trampoline
    ctx.0[12] = kstack_top; // sp = kernel stack top
    ctx.0[0] = spec.entry; // x19 = user entry PC
    ctx.0[1] = spec.stack_top; // x20 = user stack pointer
    ctx.0[2] = spec.aspace.0; // x21 = TTBR0 value
    ctx.0[3] = spec.arg as u64; // x22 = startup argument
    ctx
}

/// Trampoline that sets up arguments and calls the thread entry point.
/// Called when a new thread is first scheduled via context_switch.
#[unsafe(naked)]
unsafe extern "C" fn trampoline() -> ! {
    naked_asm!(
        "mov x0, x20", // arg is in x20
        "br x19",      // entry is in x19, jump (not call since it's noreturn)
    );
}

/// User-mode trampoline: transitions from EL1 (kernel) to EL0 (user).
///
/// Called via `context_switch` when a newly spawned user thread is first
/// scheduled. The callee-saved registers carry the task parameters set up
/// by `init_user_context`:
///   x19 = user entry PC
///   x20 = user stack pointer (SP_EL0)
///   x21 = TTBR0 value (user address space)
///   x22 = startup argument (forwarded to x0 in user mode)
///
/// SPSR_EL1 is set for EL0t (user mode) with all interrupts unmasked.
#[unsafe(naked)]
unsafe extern "C" fn user_trampoline() -> ! {
    naked_asm!(
        // Save current kernel SP, then switch to SP_EL1 mode so we can
        // write SP_EL0 without clobbering the kernel stack pointer.
        "mov x9, sp",
        "msr spsel, #1",
        "mov sp, x9",
        // Set the user-mode register state.
        "msr sp_el0,   x20",  // user stack pointer
        "msr elr_el1,  x19",  // user entry PC
        "msr ttbr0_el1, x21", // user address space
        "isb",                // ensure TTBR0 is visible before eret
        // SPSR_EL1 = 0 → EL0t mode, DAIF bits 0 (interrupts unmasked)
        "msr spsr_el1, xzr",
        // Forward startup argument to x0; clear all other general registers.
        "mov x0,  x22",
        "mov x1,  xzr",
        "mov x2,  xzr",
        "mov x3,  xzr",
        "mov x4,  xzr",
        "mov x5,  xzr",
        "mov x6,  xzr",
        "mov x7,  xzr",
        "mov x8,  xzr",
        "mov x9,  xzr",
        "mov x10, xzr",
        "mov x11, xzr",
        "mov x12, xzr",
        "mov x13, xzr",
        "mov x14, xzr",
        "mov x15, xzr",
        "mov x16, xzr",
        "mov x17, xzr",
        "mov x18, xzr",
        "mov x19, xzr",
        "mov x20, xzr",
        "mov x21, xzr",
        "mov x22, xzr",
        "mov x23, xzr",
        "mov x24, xzr",
        "mov x25, xzr",
        "mov x26, xzr",
        "mov x27, xzr",
        "mov x28, xzr",
        "mov x29, xzr",
        "mov x30, xzr",
        "eret",
    );
}

pub unsafe fn switch(from: &mut AArch64Context, to: &AArch64Context) {
    unsafe { context_switch(from.0.as_mut_ptr(), to.0.as_ptr()) };
}
