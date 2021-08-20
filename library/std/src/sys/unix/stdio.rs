use crate::io::{self, IoSlice, IoSliceMut};
use crate::mem::ManuallyDrop;
use crate::os::unix::io::{AsFd, BorrowedFd, FromRawFd};
use crate::sys::fd::FileDesc;

pub struct Stdin(());
pub struct Stdout(());
pub struct Stderr(());

impl Stdin {
    pub const fn new() -> Stdin {
        Stdin(())
    }
}

impl io::Read for Stdin {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        unsafe { ManuallyDrop::new(FileDesc::from_raw_fd(libc::STDIN_FILENO)).read(buf) }
    }

    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        unsafe { ManuallyDrop::new(FileDesc::from_raw_fd(libc::STDIN_FILENO)).read_vectored(bufs) }
    }

    #[inline]
    fn is_read_vectored(&self) -> bool {
        true
    }
}

impl Stdout {
    pub const fn new() -> Stdout {
        Stdout(())
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        unsafe { ManuallyDrop::new(FileDesc::from_raw_fd(libc::STDOUT_FILENO)).write(buf) }
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        unsafe {
            ManuallyDrop::new(FileDesc::from_raw_fd(libc::STDOUT_FILENO)).write_vectored(bufs)
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
        Stderr(())
    }
}

impl io::Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        unsafe { ManuallyDrop::new(FileDesc::from_raw_fd(libc::STDERR_FILENO)).write(buf) }
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        unsafe {
            ManuallyDrop::new(FileDesc::from_raw_fd(libc::STDERR_FILENO)).write_vectored(bufs)
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

pub fn is_ebadf(err: &io::Error) -> bool {
    err.raw_os_error() == Some(libc::EBADF as i32)
}

pub const STDIN_BUF_SIZE: usize = crate::sys_common::io::DEFAULT_BUF_SIZE;

pub fn panic_output() -> Option<impl io::Write> {
    Some(Stderr::new())
}

#[unstable(feature = "io_safety", issue = "87074")]
impl AsFd for io::Stdin {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        unsafe { BorrowedFd::borrow_raw_fd(libc::STDIN_FILENO) }
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl<'a> AsFd for io::StdinLock<'a> {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        unsafe { BorrowedFd::borrow_raw_fd(libc::STDIN_FILENO) }
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl AsFd for io::Stdout {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        unsafe { BorrowedFd::borrow_raw_fd(libc::STDOUT_FILENO) }
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl<'a> AsFd for io::StdoutLock<'a> {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        unsafe { BorrowedFd::borrow_raw_fd(libc::STDOUT_FILENO) }
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl AsFd for io::Stderr {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        unsafe { BorrowedFd::borrow_raw_fd(libc::STDERR_FILENO) }
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl<'a> AsFd for io::StderrLock<'a> {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        unsafe { BorrowedFd::borrow_raw_fd(libc::STDERR_FILENO) }
    }
}
