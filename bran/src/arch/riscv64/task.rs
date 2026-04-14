use super::paging::RISCV64AddressSpace;
use core::arch::asm;
use kernel::UserTaskSpec;

#[derive(Copy, Clone, Default)]
pub struct RISCV64Context(pub [u64; 14]); // ra, sp, s0-s11

#[unsafe(no_mangle)]
pub unsafe extern "C" fn context_switch(_old: *mut u64, _new: *const u64) {
    unsafe {
        asm!(
            "sd ra, 0(a0)",
            "sd sp, 8(a0)",
            "sd s0, 16(a0)",
            "sd s1, 24(a0)",
            "sd s2, 32(a0)",
            "sd s3, 40(a0)",
            "sd s4, 48(a0)",
            "sd s5, 56(a0)",
            "sd s6, 64(a0)",
            "sd s7, 72(a0)",
            "sd s8, 80(a0)",
            "sd s9, 88(a0)",
            "sd s10, 96(a0)",
            "sd s11, 104(a0)",
            "ld ra, 0(a1)",
            "ld sp, 8(a1)",
            "ld s0, 16(a1)",
            "ld s1, 24(a1)",
            "ld s2, 32(a1)",
            "ld s3, 40(a1)",
            "ld s4, 48(a1)",
            "ld s5, 56(a1)",
            "ld s6, 64(a1)",
            "ld s7, 72(a1)",
            "ld s8, 80(a1)",
            "ld s9, 88(a1)",
            "ld s10, 96(a1)",
            "ld s11, 104(a1)",
            "ret",
            options(noreturn)
        );
    }
}

pub fn init_kernel_context(
    entry: extern "C" fn(usize) -> !,
    stack_top: u64,
    arg: usize,
) -> RISCV64Context {
    let mut ctx = RISCV64Context::default();
    ctx.0[0] = trampoline as *const () as u64; // ra
    ctx.0[1] = stack_top; // sp
    ctx.0[2] = entry as *const () as u64; // s0
    ctx.0[3] = arg as u64; // s1
    ctx
}

pub fn init_user_context(
    spec: UserTaskSpec<RISCV64AddressSpace>,
    kstack_top: u64,
) -> RISCV64Context {
    let mut ctx = RISCV64Context::default();
    // When context_switch restores this context, ra points to
    // user_trampoline which performs the S-mode → U-mode transition via sret.
    // Callee-saved registers carry the user task parameters:
    //   s0 = user entry PC    (→ sepc)
    //   s1 = startup argument  (→ a0 in user mode)
    //   s2 = satp value        (user address space)
    //   s3 = user stack pointer (→ sp in user mode)
    ctx.0[0] = user_trampoline as *const () as u64; // ra = user_trampoline
    ctx.0[1] = kstack_top; // sp = kernel stack top
    ctx.0[2] = spec.entry; // s0 = user entry PC
    ctx.0[3] = spec.arg as u64; // s1 = startup argument
    ctx.0[4] = spec.aspace.0; // s2 = satp value
    ctx.0[5] = spec.stack_top; // s3 = user stack pointer
    ctx
}

#[unsafe(no_mangle)]
unsafe extern "C" fn trampoline() -> ! {
    unsafe {
        asm!("mv a0, s1", "jalr s0", "ebreak", options(noreturn));
    }
}

/// User-mode trampoline: transitions from S-mode (supervisor) to U-mode (user).
///
/// Called via `context_switch` when a newly spawned user thread is first
/// scheduled. Callee-saved registers carry the task parameters set up by
/// `init_user_context`:
///   s0 = user entry PC  (→ sepc)
///   s1 = startup arg    (→ a0 in user mode)
///   s2 = satp value     (user address space)
///   s3 = user SP        (→ sp in user mode)
///
/// sstatus.SPP is cleared (U-mode) and SPIE is cleared (interrupts masked
/// in user mode on entry, consistent with `enter_user`).
#[unsafe(no_mangle)]
unsafe extern "C" fn user_trampoline() -> ! {
    unsafe {
        asm!(
            // Read sstatus and configure for U-mode sret
            "csrr t0, sstatus",
            "andi t0, t0, -0x121", // clear SPP (bit 8) and SPIE (bit 5)
            "csrw sstatus, t0",
            // Write address space (satp) and fence
            "csrw satp, s2",
            "sfence.vma",
            // Write user entry PC
            "csrw sepc, s0",
            // Set up user stack and argument
            "mv sp, s3",
            "mv a0, s1",
            // Zero caller-saved registers before entering user mode
            "li ra, 0",
            "li a1, 0",
            "li a2, 0",
            "li a3, 0",
            "li a4, 0",
            "li a5, 0",
            "li a6, 0",
            "li a7, 0",
            "li t0, 0",
            "li t1, 0",
            "li t2, 0",
            "li t3, 0",
            "li t4, 0",
            "li t5, 0",
            "li t6, 0",
            // Zero callee-saved regs (s0-s11) to avoid leaking kernel data
            "li s0, 0",
            "li s1, 0",
            "li s2, 0",
            "li s3, 0",
            "li s4, 0",
            "li s5, 0",
            "li s6, 0",
            "li s7, 0",
            "li s8, 0",
            "li s9, 0",
            "li s10, 0",
            "li s11, 0",
            "sret",
            options(noreturn)
        );
    }
}

pub unsafe fn switch(from: &mut RISCV64Context, to: &RISCV64Context) {
    unsafe { context_switch(from.0.as_mut_ptr(), to.0.as_ptr()) };
}
