use core::arch::global_asm;

// AArch64 Syscall Entry
// Per ABI:
// X8 = Syscall Number
// X0..X5 = Args
// Ret = X0

global_asm!(
    r#"
.section .text
.global handle_sync_exception
// Note: We need to hook into the exception vector table.
// Usually this is done in trap.rs where specific offsets jump to handlers.
// If handle_sync_exception is already defined there, we should modify it.
// Assuming we are called from trap handler dispatch.
"#
);

// Actually, on AArch64, syscalls triggers a Synchronous Exception at EL1.
// We need to check ESR_EL1 to see if it's an SVC instruction.
// If so, we handle it.

use super::trap::UserTrapFrame;
use kernel::syscall::dispatch;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn handle_syscall(tf: &mut UserTrapFrame) {
    // AArch64 SVC puts syscall number in X8
    let n = tf.regs[8] as usize; // x8 is index 8 (x[0] is index 0)

    // In UserTrapFrame, x is usually [u64; 31] or 32?
    // Let's assume standard layout.

    // Args in x0..x5
    let a0 = tf.regs[0] as usize;
    let a1 = tf.regs[1] as usize;
    let a2 = tf.regs[2] as usize;
    let a3 = tf.regs[3] as usize;
    let a4 = tf.regs[4] as usize;
    let a5 = tf.regs[5] as usize;

    // Dispatch
    let ret = dispatch(n, [a0, a1, a2, a3, a4, a5]);

    // Return value in X0
    tf.regs[0] = ret as u64;
}
