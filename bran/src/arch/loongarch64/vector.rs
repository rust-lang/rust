use super::trap::UserTrapFrame;
use core::arch::{asm, global_asm};
use kernel::syscall::dispatch;

pub unsafe fn init() {
    unsafe extern "C" {
        fn trap_entry();
    }
    let addr = trap_entry as *const () as usize;
    // Set EENTRY (CSR 0xC)
    unsafe {
        asm!("csrwr {}, 0xC", in(reg) addr);
    }
}

global_asm!(
    r#"
.section .text
.global trap_entry
.balign 4
trap_entry:
    // Save $t0 to KS1 (SCRATCH 1 - CSR 0x31) temporarily so we can use it
    csrwr $t0, 0x31

    // Check PRMD (CSR 0x1) to see if we came from User or Kernel
    csrrd $t0, 0x1
    
    // Extract PPLv (bits 1:0). 3=User, 0=Kernel.
    andi $t0, $t0, 3
    
    // Check if PPLv == 3 (User)
    // We can't easily branch without clobbering more regs or using tricky logic.
    // Instead, let's just do a conditional branch.
    // We need to be careful about what registers we use.
    // $t0 is dirty (holds PPLv). KS1 holds original $t0.
    
    addi.d $t0, $t0, -3
    bnez $t0, 1f 
    
    // -- USER MODE TRAP --
    // Swap SP with KS0 (User Stack <-> Kernel Stack)
    // Current SP is User SP. KS0 is Kernel Stack.
    csrwr $sp, 0x30
    // Now SP is Kernel Stack.
    
    b 2f

1:
    // -- KERNEL MODE TRAP --
    // SP is already Kernel Stack. Do nothing.

2:
    // Restore original $t0 from KS1 to $t0 (for saving)
    csrrd $t0, 0x31
    
    // Alloc frame
    addi.d $sp, $sp, -288 
    
    // Save regs (r1..r31)
    st.d $r1, $sp, 0
    st.d $r2, $sp, 8  
    
    // For r3 (SP), we need to save the value it had BEFORE we alloc'd frame.
    // If from User, that's the User SP (now in KS0).
    // If from Kernel, that's (current SP + 288).
    // We'll calculate it and overwrite slot 16 later.
    
    st.d $r4, $sp, 24
    st.d $r5, $sp, 32
    st.d $r6, $sp, 40
    st.d $r7, $sp, 48
    st.d $r8, $sp, 56
    st.d $r9, $sp, 64
    st.d $r10, $sp, 72
    st.d $r11, $sp, 80
    st.d $r12, $sp, 88
    st.d $r13, $sp, 96
    st.d $r14, $sp, 104
    st.d $r15, $sp, 112
    st.d $r16, $sp, 120
    st.d $r17, $sp, 128
    st.d $r18, $sp, 136
    st.d $r19, $sp, 144
    st.d $r20, $sp, 152
    st.d $r21, $sp, 160
    st.d $r22, $sp, 168
    st.d $r23, $sp, 176
    st.d $r24, $sp, 184
    st.d $r25, $sp, 192
    st.d $r26, $sp, 200
    st.d $r27, $sp, 208
    st.d $r28, $sp, 216
    st.d $r29, $sp, 224
    st.d $r30, $sp, 232
    st.d $r31, $sp, 240
    
    // Handle SP saving
    // Re-check PRMD
    csrrd $t0, 0x1
    andi $t0, $t0, 3
    addi.d $t0, $t0, -3
    bnez $t0, 3f
    
    // User Mode: SP is in KS0
    csrrd $t0, 0x30
    st.d $t0, $sp, 16 
    b 4f

3:
    // Kernel Mode: SP was `sp + 288`
    addi.d $t0, $sp, 288
    st.d $t0, $sp, 16

4:
    // Save CSRs
    // ERA (0x6)
    csrrd $t0, 0x6
    st.d $t0, $sp, 248
    
    // PRMD (0x1)
    csrrd $t0, 0x1
    st.d $t0, $sp, 256
    
    // BADV (0x7)
    csrrd $t0, 0x7
    st.d $t0, $sp, 264
    
    // ESTAT (0x5)
    csrrd $t0, 0x5
    st.d $t0, $sp, 272
    
    // Call Rust
    move $a0, $sp
    bl rust_trap_handler
    
    // Restore CSRs
    ld.d $t0, $sp, 248
    csrwr $t0, 0x6
    
    ld.d $t0, $sp, 256
    csrwr $t0, 0x1
    
    // Restore regs
    ld.d $r1, $sp, 0
    ld.d $r2, $sp, 8
    // Skip r3 (SP)
    ld.d $r4, $sp, 24
    ld.d $r5, $sp, 32
    ld.d $r6, $sp, 40
    ld.d $r7, $sp, 48
    ld.d $r8, $sp, 56
    ld.d $r9, $sp, 64
    ld.d $r10, $sp, 72
    ld.d $r11, $sp, 80
    ld.d $r12, $sp, 88
    ld.d $r13, $sp, 96
    ld.d $r14, $sp, 104
    ld.d $r15, $sp, 112
    ld.d $r16, $sp, 120
    ld.d $r17, $sp, 128
    ld.d $r18, $sp, 136
    ld.d $r19, $sp, 144
    ld.d $r20, $sp, 152
    ld.d $r21, $sp, 160
    ld.d $r22, $sp, 168
    ld.d $r23, $sp, 176
    ld.d $r24, $sp, 184
    ld.d $r25, $sp, 192
    ld.d $r26, $sp, 200
    ld.d $r27, $sp, 208
    ld.d $r28, $sp, 216
    ld.d $r29, $sp, 224
    ld.d $r30, $sp, 232
    ld.d $r31, $sp, 240
    
    // Decide if we restore User SP or just dealloc
    ld.d $t0, $sp, 256 // PRMD
    andi $t0, $t0, 3 // PPLv
    addi.d $t0, $t0, -3
    bnez $t0, 5f
    
    // Returning to User:
    // Restore User SP from slot 16 to KS0
    ld.d $t0, $sp, 16
    csrwr $t0, 0x30
    
    addi.d $sp, $sp, 288
    
    // Swap, so $sp becomes User SP, and KS0 becomes Kernel Stack
    csrwr $sp, 0x30
    
    ertn

5:
    // Returning to Kernel:
    // Just dealloc. SP is maintained.
    addi.d $sp, $sp, 288
    ertn
"#
);

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_trap_handler(tf: &mut UserTrapFrame) {
    let estat = tf.estat;
    let ecode = (estat >> 16) & 0x3F;
    let subcode = estat & 0xFFFF;

    if ecode == 0xB {
        // SYSCALL
        // Syscall num in A7 (R11).
        // R11 is index 10 (regs[10]). (r1=index 0)
        // Regs: r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11.
        // Index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.
        let n = tf.regs[10];

        // Args: A0..A5
        // A0=r4 (index 3)
        // A1=r5 (index 4)
        // A2=r6 (index 5)
        // A3=r7 (index 6)
        // A4=r8 (index 7)
        // A5=r9 (index 8)

        let a0 = tf.regs[3];
        let a1 = tf.regs[4];
        let a2 = tf.regs[5];
        let a3 = tf.regs[6];
        let a4 = tf.regs[7];
        let a5 = tf.regs[8];

        let ret = dispatch(n, [a0, a1, a2, a3, a4, a5]);

        // Return value in A0 (r4, index 3)
        tf.regs[3] = ret as usize;

        // Advance ERA by 4 (instruction size)
        tf.era += 4;
    } else {
        // Interrupts: ecode == 0, interrupt source in ESTAT IS field (bits 12:0).
        // IS[10] = TI (Timer Interrupt), IS[11] = IPI.
        // Routing both through the same path would conflate tick-counting with
        // cross-CPU reschedule requests (see issue #498 for the x86_64 fix).
        let isr = estat & 0x1FFF;
        if isr & (1 << 10) != 0 {
            // Timer interrupt (TI): advance time accounting and reschedule if needed.
            kernel::sched::on_tick::<crate::arch::CurrentRuntime>();
        } else if isr & (1 << 11) != 0 {
            // IPI interrupt: cross-CPU reschedule request.
            // Must NOT increment TICK_COUNT - only trigger a reschedule.
            kernel::sched::on_resched_ipi::<crate::arch::CurrentRuntime>();
        } else if isr != 0 {
            // Other hardware interrupt - no scheduler action needed.
            kernel::kprintln!(
                "Unhandled LoongArch interrupt: ESTAT={:#x} ISR={:#x}",
                estat,
                isr
            );
        } else {
            kernel::kprintln!(
                "Unexpected LoongArch trap: ESTAT={:#x} ECODE={:#x} SUBCODE={:#x} ERA={:#x} BADV={:#x}",
                estat,
                ecode,
                subcode,
                tf.era,
                tf.badv
            );
            loop {}
        }
    }
}
