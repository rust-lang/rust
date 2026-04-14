//! ThingOS `is_terminal` implementation.
//!
//! Uses the `SYS_ISATTY` syscall.

use crate::os::fd::{AsFd, AsRawFd};
use crate::sys::pal::common::{SYS_ISATTY, raw_syscall6};

pub fn is_terminal(fd: &impl AsFd) -> bool {
    let raw = fd.as_fd().as_raw_fd();
    let ret = unsafe { raw_syscall6(SYS_ISATTY, raw as u64, 0, 0, 0, 0, 0) };
    ret == 1
}
