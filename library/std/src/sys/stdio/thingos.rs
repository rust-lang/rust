//! ThingOS stdio implementation.
//!
//! Standard streams are ordinary file descriptors 0 (stdin), 1 (stdout),
//! and 2 (stderr).

use crate::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use crate::sys::pal::common::{SYS_ISATTY, SYS_READ, SYS_WRITE, cvt, raw_syscall6};
use crate::sys::{FromInner, IntoInner};
use crate::{io, process, sys};

pub const STDIN_BUF_SIZE: usize = crate::sys::io::DEFAULT_BUF_SIZE;

pub struct Stdin {}
pub struct Stdout {}
pub struct Stderr {}

impl Stdin {
    pub const fn new() -> Self {
        Self {}
    }
}

impl Stdout {
    pub const fn new() -> Self {
        Self {}
    }
}

impl Stderr {
    pub const fn new() -> Self {
        Self {}
    }
}

impl crate::sealed::Sealed for Stdin {}

impl crate::io::IsTerminal for Stdin {
    fn is_terminal(&self) -> bool {
        is_terminal_fd(0)
    }
}

impl io::Read for Stdin {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let ret = unsafe {
            raw_syscall6(SYS_READ, 0, buf.as_mut_ptr() as u64, buf.len() as u64, 0, 0, 0)
        };
        cvt(ret).map(|n| n as usize)
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let ret = unsafe {
            raw_syscall6(SYS_WRITE, 1, buf.as_ptr() as u64, buf.len() as u64, 0, 0, 0)
        };
        cvt(ret).map(|n| n as usize)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl io::Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let ret = unsafe {
            raw_syscall6(SYS_WRITE, 2, buf.as_ptr() as u64, buf.len() as u64, 0, 0, 0)
        };
        cvt(ret).map(|n| n as usize)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub fn panic_output() -> Option<impl io::Write> {
    Some(Stderr::new())
}

pub fn is_ebadf(_err: &io::Error) -> bool {
    false
}

fn is_terminal_fd(fd: u64) -> bool {
    let ret = unsafe { raw_syscall6(SYS_ISATTY, fd, 0, 0, 0, 0, 0) };
    ret == 1
}

// ── process::Stdio conversions ────────────────────────────────────────────────

#[stable(feature = "process_extensions", since = "1.2.0")]
impl FromRawFd for process::Stdio {
    #[inline]
    unsafe fn from_raw_fd(fd: RawFd) -> process::Stdio {
        let fd = unsafe { sys::fd::FileDesc::from_raw_fd(fd) };
        let io = sys::process::Stdio::Fd(fd);
        process::Stdio::from_inner(io)
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<OwnedFd> for process::Stdio {
    #[inline]
    fn from(fd: OwnedFd) -> process::Stdio {
        let fd = sys::fd::FileDesc::from_inner(fd);
        let io = sys::process::Stdio::Fd(fd);
        process::Stdio::from_inner(io)
    }
}

// ── Child stream raw-fd access ────────────────────────────────────────────────

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawFd for process::ChildStdin {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.as_inner().as_raw_fd()
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawFd for process::ChildStdout {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.as_inner().as_raw_fd()
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawFd for process::ChildStderr {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.as_inner().as_raw_fd()
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for process::ChildStdin {
    #[inline]
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_raw_fd()
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for process::ChildStdout {
    #[inline]
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_raw_fd()
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for process::ChildStderr {
    #[inline]
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_raw_fd()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsFd for process::ChildStdin {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.as_inner().as_fd()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<process::ChildStdin> for OwnedFd {
    #[inline]
    fn from(child_stdin: process::ChildStdin) -> OwnedFd {
        child_stdin.into_inner().into_inner()
    }
}

#[stable(feature = "child_stream_from_fd", since = "1.74.0")]
impl From<OwnedFd> for process::ChildStdin {
    #[inline]
    fn from(fd: OwnedFd) -> process::ChildStdin {
        let pipe = sys::process::ChildPipe::from_inner(fd);
        process::ChildStdin::from_inner(pipe)
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsFd for process::ChildStdout {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.as_inner().as_fd()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<process::ChildStdout> for OwnedFd {
    #[inline]
    fn from(child_stdout: process::ChildStdout) -> OwnedFd {
        child_stdout.into_inner().into_inner()
    }
}

#[stable(feature = "child_stream_from_fd", since = "1.74.0")]
impl From<OwnedFd> for process::ChildStdout {
    #[inline]
    fn from(fd: OwnedFd) -> process::ChildStdout {
        let pipe = sys::process::ChildPipe::from_inner(fd);
        process::ChildStdout::from_inner(pipe)
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsFd for process::ChildStderr {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.as_inner().as_fd()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<process::ChildStderr> for OwnedFd {
    #[inline]
    fn from(child_stderr: process::ChildStderr) -> OwnedFd {
        child_stderr.into_inner().into_inner()
    }
}

#[stable(feature = "child_stream_from_fd", since = "1.74.0")]
impl From<OwnedFd> for process::ChildStderr {
    #[inline]
    fn from(fd: OwnedFd) -> process::ChildStderr {
        let pipe = sys::process::ChildPipe::from_inner(fd);
        process::ChildStderr::from_inner(pipe)
    }
}
