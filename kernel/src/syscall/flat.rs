use super::dispatch;

/// Flat entry point for assembly.
///
/// The `frame_ptr` parameter receives the kernel-stack pointer of the saved
/// user trap frame (`SyscallFrame` / `UserTrapFrame`).  This pointer is used
/// for two purposes:
///
/// 1. After every syscall, `kernel_post_syscall_signal_check` is called with
///    this pointer to deliver any pending signals to the user before returning.
/// 2. `SYS_SIGRETURN` passes the frame pointer to `sys_sigreturn` so the
///    signal handler's saved register state can be restored in-place.
#[unsafe(no_mangle)]
pub extern "C" fn kernel_dispatch_flat(
    n: usize,
    a0: usize,
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
    a5: usize,
    frame_ptr: usize,
) -> isize {
    use abi::syscall::SYS_SIGRETURN;

    // SYS_SIGRETURN needs direct access to the trap frame to restore registers.
    if n as u32 == SYS_SIGRETURN {
        let result = unsafe {
            crate::signal::delivery::sys_sigreturn_inner(frame_ptr as *mut u8)
        };
        // Do NOT run signal check after sigreturn: we just restored user state.
        return match result {
            Ok(v) => v as isize,
            Err(e) => -(e as isize),
        };
    }

    let ret = dispatch(n, [a0, a1, a2, a3, a4, a5]);

    // After every syscall, check for pending signals and deliver them.
    if frame_ptr != 0 {
        unsafe {
            crate::signal::delivery::kernel_post_syscall_signal_check(frame_ptr as *mut u8);
        }
    }

    ret
}
