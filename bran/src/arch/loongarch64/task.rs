use super::paging::LoongArch64AddressSpace;
use core::arch::asm;
use kernel::UserTaskSpec;

#[derive(Copy, Clone, Default)]
pub struct LoongArch64Context(pub [u64; 12]); // ra, sp, fp, s0-s8

#[unsafe(no_mangle)]
pub unsafe extern "C" fn context_switch(_old: *mut u64, _new: *const u64) {
    unsafe {
        asm!(
            "st.d $ra, $a0, 0",
            "st.d $sp, $a0, 8",
            "st.d $fp, $a0, 16",
            "st.d $s0, $a0, 24",
            "st.d $s1, $a0, 32",
            "st.d $s2, $a0, 40",
            "st.d $s3, $a0, 48",
            "st.d $s4, $a0, 56",
            "st.d $s5, $a0, 64",
            "st.d $s6, $a0, 72",
            "st.d $s7, $a0, 80",
            "st.d $s8, $a0, 88",
            "ld.d $ra, $a1, 0",
            "ld.d $sp, $a1, 8",
            "ld.d $fp, $a1, 16",
            "ld.d $s0, $a1, 24",
            "ld.d $s1, $a1, 32",
            "ld.d $s2, $a1, 40",
            "ld.d $s3, $a1, 48",
            "ld.d $s4, $a1, 56",
            "ld.d $s5, $a1, 64",
            "ld.d $s6, $a1, 72",
            "ld.d $s7, $a1, 80",
            "ld.d $s8, $a1, 88",
            "jirl $zero, $ra, 0",
            options(noreturn)
        );
    }
}

pub fn init_kernel_context(
    entry: extern "C" fn(usize) -> !,
    stack_top: u64,
    arg: usize,
) -> LoongArch64Context {
    let mut ctx = LoongArch64Context::default();
    ctx.0[0] = trampoline as *const () as u64; // ra
    ctx.0[1] = stack_top; // sp
    ctx.0[3] = entry as *const () as u64; // s0
    ctx.0[4] = arg as u64; // s1
    ctx
}

pub fn init_user_context(
    spec: UserTaskSpec<LoongArch64AddressSpace>,
    kstack_top: u64,
) -> LoongArch64Context {
    let mut ctx = LoongArch64Context::default();
    // When context_switch restores this context, ra points to
    // user_trampoline which performs the privilege transition via ertn.
    // Callee-saved registers carry the user task parameters:
    //   s0 = user entry PC    (→ ERA CSR 0x6)
    //   s1 = startup argument  (→ a0 in user mode)
    //   s2 = PGDL value        (user address space, low half)
    //   s3 = PGDH value        (user address space, high half)
    //   s4 = user stack pointer (→ sp in user mode)
    ctx.0[0] = user_trampoline as *const () as u64; // ra = user_trampoline
    ctx.0[1] = kstack_top; // sp = kernel stack top
    ctx.0[3] = spec.entry; // s0 = user entry PC
    ctx.0[4] = spec.arg as u64; // s1 = startup argument
    ctx.0[5] = spec.aspace.pgdl; // s2 = PGDL
    ctx.0[6] = spec.aspace.pgdh; // s3 = PGDH
    ctx.0[7] = spec.stack_top; // s4 = user stack pointer
    ctx
}

#[unsafe(no_mangle)]
unsafe extern "C" fn trampoline() -> ! {
    unsafe {
        asm!(
            "or $a0, $s1, $zero",
            "jirl $zero, $s0, 0",
            "break 0",
            options(noreturn)
        );
    }
}

/// User-mode trampoline: transitions from PLV0 (kernel) to PLV3 (user).
///
/// Called via `context_switch` when a newly spawned user thread is first
/// scheduled. Callee-saved registers carry the task parameters set up by
/// `init_user_context`:
///   s0 = user entry PC    (→ ERA CSR 0x6)
///   s1 = startup argument  (→ $a0 in user mode)
///   s2 = PGDL value        (user address space, low half)
///   s3 = PGDH value        (user address space, high half)
///   s4 = user stack pointer (→ $sp in user mode)
///
/// PRMD is configured for PLV3 (user mode) with PIE cleared
/// (interrupts masked on entry, consistent with `enter_user`).
#[unsafe(no_mangle)]
unsafe extern "C" fn user_trampoline() -> ! {
    unsafe {
        asm!(
            // Load address space registers
            "csrwr $s2, 0x19", // PGDL ← s2
            "csrwr $s3, 0x1a", // PGDH ← s3
            // Configure PRMD: PLV=3 (user), clear PIE (bit 2)
            "csrrd $t0, 0x1",       // read PRMD
            "ori   $t0, $t0, 3",    // set PLV bits to 3 (user)
            "andi  $t0, $t0, 0xFB", // clear PIE (bit 2)
            "csrwr $t0, 0x1",       // write PRMD
            // Set ERA (exception return address = user entry PC)
            "csrwr $s0, 0x6",
            // Set user stack pointer and argument
            "move  $sp, $s4",
            "move  $a0, $s1",
            // Zero callee-saved regs to avoid leaking kernel data
            "move  $s0, $zero",
            "move  $s1, $zero",
            "move  $s2, $zero",
            "move  $s3, $zero",
            "move  $s4, $zero",
            "move  $s5, $zero",
            "move  $s6, $zero",
            "move  $s7, $zero",
            "move  $s8, $zero",
            // Zero caller-saved regs
            "move  $ra, $zero",
            "move  $a1, $zero",
            "move  $a2, $zero",
            "move  $a3, $zero",
            "move  $a4, $zero",
            "move  $a5, $zero",
            "move  $a6, $zero",
            "move  $a7, $zero",
            "move  $t0, $zero",
            "move  $t1, $zero",
            "move  $t2, $zero",
            "move  $t3, $zero",
            "move  $t4, $zero",
            "move  $t5, $zero",
            "move  $t6, $zero",
            "move  $t7, $zero",
            "move  $t8, $zero",
            "ertn",
            options(noreturn)
        );
    }
}

pub unsafe fn switch(from: &mut LoongArch64Context, to: &LoongArch64Context) {
    unsafe { context_switch(from.0.as_mut_ptr(), to.0.as_ptr()) };
}
