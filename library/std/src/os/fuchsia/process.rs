//! Fuchsia-specific extensions to primitives in the [`std::process`] module.
//!
//! [`std::process`]: crate::process

use crate::process;
use crate::sealed::Sealed;

#[unstable(feature = "fuchsia_exit_status", issue = "none")]
pub type zx_status_t = i32;
#[unstable(feature = "fuchsia_exit_status", issue = "none")]
pub const ZX_TASK_RETCODE_SYSCALL_KILL: zx_status_t = -1024;
#[unstable(feature = "fuchsia_exit_status", issue = "none")]
pub const ZX_TASK_RETCODE_OOM_KILL: zx_status_t = -1025;
#[unstable(feature = "fuchsia_exit_status", issue = "none")]
pub const ZX_TASK_RETCODE_POLICY_KILL: zx_status_t = -1026;
#[unstable(feature = "fuchsia_exit_status", issue = "none")]
pub const ZX_TASK_RETCODE_VDSO_KILL: zx_status_t = -1027;
/// On Zircon (the Fuchsia kernel), an abort from userspace calls the
/// LLVM implementation of __builtin_trap(), e.g., ud2 on x86, which
/// raises a kernel exception. If a userspace process does not
/// otherwise arrange exception handling, the kernel kills the process
/// with this return code.
#[unstable(feature = "fuchsia_exit_status", issue = "none")]
pub const ZX_TASK_RETCODE_EXCEPTION_KILL: zx_status_t = -1028;
#[unstable(feature = "fuchsia_exit_status", issue = "none")]
pub const ZX_TASK_RETCODE_CRITICAL_PROCESS_KILL: zx_status_t = -1029;

#[unstable(feature = "fuchsia_exit_status", issue = "none")]
pub trait ExitStatusExt: Sealed {
    /// If the task was killed, returns the `ZX_TASK_RETCODE_*`.
    #[must_use]
    fn task_retcode(&self) -> Option<i32>;
}

#[unstable(feature = "fuchsia_exit_status", issue = "none")]
impl ExitStatusExt for process::ExitStatus {
    fn task_retcode(&self) -> Option<i32> {
        self.code().and_then(|code| {
            if code == ZX_TASK_RETCODE_SYSCALL_KILL
                || code == ZX_TASK_RETCODE_OOM_KILL
                || code == ZX_TASK_RETCODE_POLICY_KILL
                || code == ZX_TASK_RETCODE_VDSO_KILL
                || code == ZX_TASK_RETCODE_EXCEPTION_KILL
                || code == ZX_TASK_RETCODE_CRITICAL_PROCESS_KILL
            {
                Some(code)
            } else {
                None
            }
        })
    }
}
