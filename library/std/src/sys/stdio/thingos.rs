//! ThingOS stdio implementation.
//!
//! File descriptors 0, 1, 2 are pre-populated by the kernel at spawn time
//! (see kernel/src/sched/spawn.rs `setup_stdio_fds`).
//!
//! Reads/writes go through `SYS_READ` (0x1400) and `SYS_WRITE` (0x1401)
//! which are the basic fd-based I/O syscalls that work for all fds including
//! the pre-populated stdio fds.
//!
//! `STDIN_BUF_SIZE` is set to 1 so that line-buffered reads behave correctly
//! on a serial/console input that delivers bytes one at a time.

use crate::io;

// Syscall numbers (abi/src/numbers.rs)
const SYS_READ: u32 = 0x1400;
const SYS_WRITE: u32 = 0x1401;

#[inline(always)]
unsafe fn raw_syscall6(
    n: u32,
    a0: usize,
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
    a5: usize,
) -> isize {
    unsafe { crate::sys::pal::raw_syscall6(n, a0, a1, a2, a3, a4, a5) }
}

pub struct Stdin;
pub struct Stdout;
pub struct Stderr;

impl Stdin {
    pub const fn new() -> Self {
        Stdin
    }
}

impl io::Read for Stdin {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let ret = unsafe {
            raw_syscall6(SYS_READ, 0, buf.as_mut_ptr() as usize, buf.len(), 0, 0, 0)
        };
        if ret < 0 {
            Err(io::Error::from_raw_os_error((-ret) as i32))
        } else {
            Ok(ret as usize)
        }
    }
}

impl Stdout {
    pub const fn new() -> Self {
        Stdout
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let ret = unsafe {
            raw_syscall6(SYS_WRITE, 1, buf.as_ptr() as usize, buf.len(), 0, 0, 0)
        };
        if ret < 0 {
            Err(io::Error::from_raw_os_error((-ret) as i32))
        } else {
            Ok(ret as usize)
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Stderr {
    pub const fn new() -> Self {
        Stderr
    }
}

impl io::Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let ret = unsafe {
            raw_syscall6(SYS_WRITE, 2, buf.as_ptr() as usize, buf.len(), 0, 0, 0)
        };
        if ret < 0 {
            Err(io::Error::from_raw_os_error((-ret) as i32))
        } else {
            Ok(ret as usize)
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub const STDIN_BUF_SIZE: usize = 1;

pub fn is_ebadf(err: &io::Error) -> bool {
    err.raw_os_error() == Some(9) // EBADF
}

pub fn panic_output() -> Option<impl io::Write> {
    Some(Stderr::new())
}
