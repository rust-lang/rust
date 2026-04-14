use super::trap::UserTrapFrame;
use core::arch::{asm, global_asm};
use kernel::syscall::dispatch;

pub unsafe fn init() {
    unsafe extern "C" {
        fn trap_entry();
    }
    // Set stvec to trap_entry (Direct mode, bit 0 = 0)
    let addr = trap_entry as *const () as usize;
    // ensure alignment (4 bytes)
    assert!(addr & 3 == 0);
    unsafe {
        asm!("csrw stvec, {}", in(reg) addr);
        asm!("csrw sscratch, x0"); // Initialize sscratch to 0 for kernel detection
    }
}

global_asm!(
    r#"
.section .text
.global trap_entry
.balign 4
trap_entry:
    // Check if coming from Kernel or User
    // If coming from User, sscratch holds Kernel Stack.
    // If coming from Kernel, sscratch holds 0 (convention).
    csrrw sp, sscratch, sp
    bnez sp, 1f

    // --- Came from Kernel Mode ---
    csrrw sp, sscratch, sp   // Restore SP (it was 0 or invalid)
    // sp is valid Kernel Stack.
    addi sp, sp, -288
    
    // Save partial registers to use tmps
    sd x1, 0(sp)   // ra
    sd x3, 16(sp)  // gp
    
    // Save Kernel SP (original value was sp + 288)
    addi t0, sp, 288
    sd t0, 8(sp)   // x2/sp
    
    j 2f

1:  // --- Came from User Mode ---
    // sp is now KStack. sscratch is User Stack.
    addi sp, sp, -288
    
    sd x1, 0(sp)
    sd x3, 16(sp)
    
    // Save User SP (from sscratch)
    csrr t0, sscratch
    sd t0, 8(sp)

2:  // --- Common Saving ---
    sd x4, 24(sp)
    sd x5, 32(sp)
    sd x6, 40(sp)
    sd x7, 48(sp)
    sd x8, 56(sp)
    sd x9, 64(sp)
    sd x10, 72(sp)
    sd x11, 80(sp)
    sd x12, 88(sp)
    sd x13, 96(sp)
    sd x14, 104(sp)
    sd x15, 112(sp)
    sd x16, 120(sp)
    sd x17, 128(sp)
    sd x18, 136(sp)
    sd x19, 144(sp)
    sd x20, 152(sp)
    sd x21, 160(sp)
    sd x22, 168(sp)
    sd x23, 176(sp)
    sd x24, 184(sp)
    sd x25, 192(sp)
    sd x26, 200(sp)
    sd x27, 208(sp)
    sd x28, 216(sp)
    sd x29, 224(sp)
    sd x30, 232(sp)
    sd x31, 240(sp)

    // Save CSRs
    csrr t0, sstatus
    sd t0, 248(sp)
    
    csrr t0, sepc
    sd t0, 256(sp)
    
    csrr t0, stval
    sd t0, 264(sp)
    
    csrr t0, scause
    sd t0, 272(sp)
    
    // Call handler(tf)
    mv a0, sp
    call rust_trap_handler
    
    // Restore
    ld t0, 248(sp)
    csrw sstatus, t0
    
    ld t0, 256(sp)
    csrw sepc, t0
    
    ld x1, 0(sp)
    ld x3, 16(sp)
    ld x4, 24(sp)
    ld x5, 32(sp)
    ld x6, 40(sp)
    ld x7, 48(sp)
    ld x8, 56(sp)
    ld x9, 64(sp)
    ld x10, 72(sp)
    ld x11, 80(sp)
    ld x12, 88(sp)
    ld x13, 96(sp)
    ld x14, 104(sp)
    ld x15, 112(sp)
    ld x16, 120(sp)
    ld x17, 128(sp)
    ld x18, 136(sp)
    ld x19, 144(sp)
    ld x20, 152(sp)
    ld x21, 160(sp)
    ld x22, 168(sp)
    ld x23, 176(sp)
    ld x24, 184(sp)
    ld x25, 192(sp)
    ld x26, 200(sp)
    ld x27, 208(sp)
    ld x28, 216(sp)
    ld x29, 224(sp)
    ld x30, 232(sp)
    ld x31, 240(sp)
    
    // Check if we need to return to User or Kernel
    // We check Previous Mode in Supervisor Status (SPP bit 8).
    // If SPP=1 (Supervisor), we return to Kernel.
    // If SPP=0 (User), we return to User.
    
    ld t0, 248(sp) // Load sstatus again (it might be modified by handler, but we restored it to CSR)
    // Actually we should read from CSR or saved value. Saved value is reliable.
    
    // Check SPP bit (bit 8)
    li t1, (1 << 8)
    and t1, t0, t1
    bnez t1, 3f
    
    // --- Return to User ---
    // Restore User SP (x2) to sscratch
    ld t0, 8(sp)
    csrw sscratch, t0
    
    addi sp, sp, 288
    // Swap sp and sscratch to restore User Stack
    csrrw sp, sscratch, sp
    sret

3:  // --- Return to Kernel ---
    // Restore Kernel SP (x2) directly to sp?
    // Wait, we are ON the kernel stack.
    // sp points to the frame.
    // We want sp to be (sp + 288).
    // But we also need to restore x2 (which IS sp).
    // The saved x2 IS (old_sp).
    // So loading x2 from stack will restore sp!
    ld x2, 8(sp)
    sret
"#
);

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_trap_handler(tf: &mut UserTrapFrame) {
    let scause = tf.scause;
    let is_interrupt = (scause >> 63) != 0;
    let code = scause & 0x7FFFFFFFFFFFFFFF;

    if is_interrupt {
        // Distinguish timer (STI, code=5) from software IPI (SSI, code=1).
        // Routing both through on_tick() would conflate tick-counting with
        // cross-CPU reschedule requests (see issue #498 for the x86_64 fix).
        match code {
            5 => {
                // S-mode Timer Interrupt (STIP): advance time accounting and
                // trigger a reschedule if needed.
                kernel::sched::on_tick::<crate::arch::CurrentRuntime>();
            }
            1 => {
                // S-mode Software Interrupt (SSIP): used as the reschedule IPI.
                // Must NOT increment TICK_COUNT - only trigger a reschedule.
                kernel::sched::on_resched_ipi::<crate::arch::CurrentRuntime>();
            }
            _ => {
                // Other interrupts (e.g. external, SEIP=9): no scheduler action.
                kernel::kprintln!(
                    "Unhandled riscv64 interrupt: scause={:#x} code={:#x}",
                    scause,
                    code
                );
            }
        }
    } else {
        match code {
            8 => {
                // User mode ecall
                // Syscall
                // A7 is syscall num.
                // A7 is x17.
                // regs[0]=x1 ... regs[16]=x17.
                // So A7 is at index 16.
                let n = tf.regs[16];

                // Args: A0..A5
                // A0=x10 -> index 9
                // A1=x11 -> index 10
                // A2=x12 -> index 11
                // A3=x13 -> index 12
                // A4=x14 -> index 13
                // A5=x15 -> index 14

                let a0 = tf.regs[9];
                let a1 = tf.regs[10];
                let a2 = tf.regs[11];
                let a3 = tf.regs[12];
                let a4 = tf.regs[13];
                let a5 = tf.regs[14];

                let ret = dispatch(n, [a0, a1, a2, a3, a4, a5]);

                // Return value in A0 (x10, index 9)
                tf.regs[9] = ret as usize;

                // Advance SEPC by 4 (size of ecall)
                tf.sepc += 4;
            }
            _ => {
                // Panic or loop
                kernel::kprintln!(
                    "Unexpected trap: scause={:x} stval={:x} sepc={:x}",
                    scause,
                    tf.stval,
                    tf.sepc
                );
                loop {}
            }
        }
    }
}
