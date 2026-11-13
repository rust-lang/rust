#[expect(dead_code)]
#[path = "unsupported.rs"]
mod unsupported_stdio;

use crate::cmp;
use crate::io::{self, IoSlice};

pub type Stdin = unsupported_stdio::Stdin;
pub struct Stdout;
pub struct Stderr;

impl Stdout {
    pub const fn new() -> Stdout {
        Stdout
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        write(libc::STDOUT_FILENO, buf)
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        write_vectored(libc::STDOUT_FILENO, bufs)
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        true
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Stderr {
    pub const fn new() -> Stderr {
        Stderr
    }
}

impl io::Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        write(libc::STDERR_FILENO, buf)
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        write_vectored(libc::STDERR_FILENO, bufs)
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        true
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub const STDIN_BUF_SIZE: usize = unsupported_stdio::STDIN_BUF_SIZE;

pub fn is_ebadf(_err: &io::Error) -> bool {
    true
}

pub fn panic_output() -> Option<impl io::Write> {
    Some(Stderr)
}

fn write(fd: i32, buf: &[u8]) -> io::Result<usize> {
    let iov = libc::iovec { iov_base: buf.as_ptr() as *mut _, iov_len: buf.len() };
    // SAFETY: syscall, safe arguments.
    let ret = unsafe { libc::writev(fd, &iov, 1) };
    // This check includes ret < 0, since the length is at most isize::MAX.
    if ret as usize > iov.iov_len {
        return Err(io::Error::last_os_error());
    }
    Ok(ret as usize)
}

fn write_vectored(fd: i32, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
    let iov = bufs.as_ptr() as *const libc::iovec;
    let len = cmp::min(bufs.len(), libc::c_int::MAX as usize) as libc::c_int;
    // SAFETY: syscall, safe arguments.
    let ret = unsafe { libc::writev(fd, iov, len) };
    if ret < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(ret as usize)
}
