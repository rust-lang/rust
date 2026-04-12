//! ThingOS anonymous pipe implementation.
//!
//! Uses the `SYS_PIPE` system call which returns two file descriptors:
//! index 0 = read end, index 1 = write end.

use crate::io;
use crate::os::fd::{FromRawFd, OwnedFd};
use crate::sys::fd::FileDesc;
use crate::sys::pal::common::{SYS_PIPE, cvt, raw_syscall6};

pub type Pipe = FileDesc;

pub fn pipe() -> io::Result<(Pipe, Pipe)> {
    let mut fds = [0i32; 2];
    let ret = unsafe {
        raw_syscall6(SYS_PIPE, fds.as_mut_ptr() as u64, 0, 0, 0, 0, 0)
    };
    cvt(ret)?;
    // SAFETY: the kernel just filled `fds` with two valid, owned file descriptors.
    unsafe {
        let read = FileDesc::from_inner(OwnedFd::from_raw_fd(fds[0]));
        let write = FileDesc::from_inner(OwnedFd::from_raw_fd(fds[1]));
        Ok((read, write))
    }
}
