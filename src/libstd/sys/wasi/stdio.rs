use crate::io::{self, IoSlice, IoSliceMut};
use crate::mem::ManuallyDrop;
use crate::sys::fd::WasiFd;

pub struct Stdin;
pub struct Stdout;
pub struct Stderr;

impl Stdin {
    pub fn new() -> io::Result<Stdin> {
        Ok(Stdin)
    }

    pub fn read(&self, data: &mut [u8]) -> io::Result<usize> {
        self.read_vectored(&mut [IoSliceMut::new(data)])
    }

    pub fn read_vectored(&self, data: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        ManuallyDrop::new(unsafe { WasiFd::from_raw(self.as_raw_fd()) }).read(data)
    }

    pub fn as_raw_fd(&self) -> u32 {
        0
    }
}

impl Stdout {
    pub fn new() -> io::Result<Stdout> {
        Ok(Stdout)
    }

    pub fn write(&self, data: &[u8]) -> io::Result<usize> {
        self.write_vectored(&[IoSlice::new(data)])
    }

    pub fn write_vectored(&self, data: &[IoSlice<'_>]) -> io::Result<usize> {
        ManuallyDrop::new(unsafe { WasiFd::from_raw(self.as_raw_fd()) }).write(data)
    }

    pub fn flush(&self) -> io::Result<()> {
        Ok(())
    }

    pub fn as_raw_fd(&self) -> u32 {
        1
    }
}

impl Stderr {
    pub fn new() -> io::Result<Stderr> {
        Ok(Stderr)
    }

    pub fn write(&self, data: &[u8]) -> io::Result<usize> {
        self.write_vectored(&[IoSlice::new(data)])
    }

    pub fn write_vectored(&self, data: &[IoSlice<'_>]) -> io::Result<usize> {
        ManuallyDrop::new(unsafe { WasiFd::from_raw(self.as_raw_fd()) }).write(data)
    }

    pub fn flush(&self) -> io::Result<()> {
        Ok(())
    }

    pub fn as_raw_fd(&self) -> u32 {
        2
    }
}

impl io::Write for Stderr {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        (&*self).write(data)
    }

    fn flush(&mut self) -> io::Result<()> {
        (&*self).flush()
    }
}

pub const STDIN_BUF_SIZE: usize = crate::sys_common::io::DEFAULT_BUF_SIZE;

pub fn is_ebadf(err: &io::Error) -> bool {
    err.raw_os_error() == Some(wasi::ERRNO_BADF.into())
}

pub fn panic_output() -> Option<impl io::Write> {
    Stderr::new().ok()
}
