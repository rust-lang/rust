use super::hermit_abi;
use crate::io;
use crate::io::{IoSlice, IoSliceMut};
use crate::mem::ManuallyDrop;
use crate::os::hermit::io::FromRawFd;
use crate::sys::fd::FileDesc;

pub struct Stdin;
pub struct Stdout;
pub struct Stderr;

impl Stdin {
    pub const fn new() -> Stdin {
        Stdin
    }
}

impl io::Read for Stdin {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        unsafe { ManuallyDrop::new(FileDesc::from_raw_fd(hermit_abi::STDIN_FILENO)).read(buf) }
    }

    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        unsafe {
            ManuallyDrop::new(FileDesc::from_raw_fd(hermit_abi::STDIN_FILENO)).read_vectored(bufs)
        }
    }

    #[inline]
    fn is_read_vectored(&self) -> bool {
        true
    }
}

impl Stdout {
    pub const fn new() -> Stdout {
        Stdout
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        unsafe { ManuallyDrop::new(FileDesc::from_raw_fd(hermit_abi::STDOUT_FILENO)).write(buf) }
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        unsafe {
            ManuallyDrop::new(FileDesc::from_raw_fd(hermit_abi::STDOUT_FILENO)).write_vectored(bufs)
        }
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
        unsafe { ManuallyDrop::new(FileDesc::from_raw_fd(hermit_abi::STDERR_FILENO)).write(buf) }
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        unsafe {
            ManuallyDrop::new(FileDesc::from_raw_fd(hermit_abi::STDERR_FILENO)).write_vectored(bufs)
        }
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        true
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub const STDIN_BUF_SIZE: usize = 128;

pub fn is_ebadf(err: &io::Error) -> bool {
    err.raw_os_error() == Some(hermit_abi::EBADF)
}

pub fn panic_output() -> Option<impl io::Write> {
    Some(Stderr::new())
}
