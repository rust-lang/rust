use core::arch::{asm, global_asm};

pub unsafe fn init() {
    unsafe extern "C" {
        static vector_table: u8;
    }
    let vbar = unsafe { &vector_table } as *const u8 as u64;
    unsafe {
        asm!(
            "msr vbar_el1, {vbar}",
            "isb",
            vbar = in(reg) vbar,
            options(nomem, nostack, preserves_flags)
        );
    }
}

global_asm!(
    r#"
.section .text
.balign 2048
.global vector_table
vector_table:
    // Current EL with SP0
    .balign 128
    b unhandled_exception // Sync
    .balign 128
    b unhandled_exception // IRQ
    .balign 128
    b unhandled_exception // FIQ
    .balign 128
    b unhandled_exception // SError

    // Current EL with SPx
    .balign 128
    b unhandled_curr_spx_sync // Sync
    .balign 128
    b unhandled_curr_spx_irq  // IRQ
    .balign 128
    b unhandled_exception // FIQ
    .balign 128
    b unhandled_exception // SError

    // Lower EL using AArch64
    .balign 128
    b handle_sync_el0     // Sync (Syscalls/Traps)
    .balign 128
    b handle_irq_el0      // IRQ from EL0
    .balign 128
    b unhandled_exception // FIQ
    .balign 128
    b unhandled_exception // SError

    // Lower EL using AArch32
    .balign 128
    b unhandled_exception
    .balign 128
    b unhandled_exception
    .balign 128
    b unhandled_exception
    .balign 128
    b unhandled_exception

unhandled_exception:
    mov x2, #0 // generic
    b unhandled_common

unhandled_curr_spx_sync:
    mov x2, #1 // Curr-SPx-Sync
    b unhandled_common

unhandled_curr_spx_irq:
    mov x2, #2 // Curr-SPx-IRQ
    b unhandled_common

unhandled_lower_irq:
    mov x2, #3 // Lower-IRQ
    b unhandled_common

unhandled_common:
    // Make stack frame
    sub sp, sp, #32
    stp x0, x1, [sp, #0]
    stp x29, x30, [sp, #16]
    
    mrs x0, esr_el1
    mrs x1, elr_el1
    // x2 has origin code
    mrs x3, spsr_el1
    
    bl unhandled_exception_rust
    
    b .

handle_sync_el0:
    // UserTrapFrame layout:
    // regs: [u64; 31] (0..240)
    // sp_el0: u64 (248)
    // elr_el1: u64 (256)
    // spsr_el1: u64 (264)
    // Total: 272 bytes. 16-align -> 272 (17 * 16).
    
    sub sp, sp, #272
    
    // Save x0-x29
    stp x0, x1, [sp, #0]
    stp x2, x3, [sp, #16]
    stp x4, x5, [sp, #32]
    stp x6, x7, [sp, #48]
    stp x8, x9, [sp, #64]
    stp x10, x11, [sp, #80]
    stp x12, x13, [sp, #96]
    stp x14, x15, [sp, #112]
    stp x16, x17, [sp, #128]
    stp x18, x19, [sp, #144]
    stp x20, x21, [sp, #160]
    stp x22, x23, [sp, #176]
    stp x24, x25, [sp, #192]
    stp x26, x27, [sp, #208]
    stp x28, x29, [sp, #224]
    
    // Save x30 (LR)
    str x30, [sp, #240]
    
    // Save SP_EL0
    mrs x9, sp_el0
    str x9, [sp, #248]
    
    // Save ELR_EL1
    mrs x10, elr_el1
    str x10, [sp, #256]
    
    // Save SPSR_EL1
    mrs x11, spsr_el1
    str x11, [sp, #264]
    
    // Read ESR to decide what kind of sync we took
    mrs x1, esr_el1
    // Call handler(tf: *mut UserTrapFrame, esr: u64)
    mov x0, sp
    bl handle_sync_el0_rust
    
    // Restore SPSR_EL1
    ldr x11, [sp, #264]
    msr spsr_el1, x11
    
    // Restore ELR_EL1
    ldr x10, [sp, #256]
    msr elr_el1, x10
    
    // Restore SP_EL0
    ldr x9, [sp, #248]
    msr sp_el0, x9
    
    // Restore x30
    ldr x30, [sp, #240]
    
    // Restore x0-x29
    ldp x28, x29, [sp, #224]
    ldp x26, x27, [sp, #208]
    ldp x24, x25, [sp, #192]
    ldp x22, x23, [sp, #176]
    ldp x20, x21, [sp, #160]
    ldp x18, x19, [sp, #144]
    ldp x16, x17, [sp, #128]
    ldp x14, x15, [sp, #112]
    ldp x12, x13, [sp, #96]
    ldp x10, x11, [sp, #80]
    ldp x8, x9, [sp, #64]
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldp x2, x3, [sp, #16]
    ldp x0, x1, [sp, #0]
    
    add sp, sp, #272
    eret

handle_irq_el0:
    // IRQ from EL0 (user mode).  Save the same frame as handle_sync_el0 so
    // that the full interrupted user context is preserved across any potential
    // context switch triggered by on_tick / on_resched_ipi.
    sub sp, sp, #272

    stp x0, x1, [sp, #0]
    stp x2, x3, [sp, #16]
    stp x4, x5, [sp, #32]
    stp x6, x7, [sp, #48]
    stp x8, x9, [sp, #64]
    stp x10, x11, [sp, #80]
    stp x12, x13, [sp, #96]
    stp x14, x15, [sp, #112]
    stp x16, x17, [sp, #128]
    stp x18, x19, [sp, #144]
    stp x20, x21, [sp, #160]
    stp x22, x23, [sp, #176]
    stp x24, x25, [sp, #192]
    stp x26, x27, [sp, #208]
    stp x28, x29, [sp, #224]
    str x30, [sp, #240]

    mrs x9, sp_el0
    str x9, [sp, #248]

    mrs x10, elr_el1
    str x10, [sp, #256]

    mrs x11, spsr_el1
    str x11, [sp, #264]

    bl handle_irq_el0_rust

    // Restore system registers
    ldr x11, [sp, #264]
    msr spsr_el1, x11
    ldr x10, [sp, #256]
    msr elr_el1, x10
    ldr x9, [sp, #248]
    msr sp_el0, x9

    ldr x30, [sp, #240]
    ldp x28, x29, [sp, #224]
    ldp x26, x27, [sp, #208]
    ldp x24, x25, [sp, #192]
    ldp x22, x23, [sp, #176]
    ldp x20, x21, [sp, #160]
    ldp x18, x19, [sp, #144]
    ldp x16, x17, [sp, #128]
    ldp x14, x15, [sp, #112]
    ldp x12, x13, [sp, #96]
    ldp x10, x11, [sp, #80]
    ldp x8, x9, [sp, #64]
    ldp x6, x7, [sp, #48]
    ldp x4, x5, [sp, #32]
    ldp x2, x3, [sp, #16]
    ldp x0, x1, [sp, #0]

    add sp, sp, #272
    eret
"#
);
