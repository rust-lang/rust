use crate::io;
use crate::io::{IoSlice, IoSliceMut};
use crate::sys::hermit::abi;

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

    pub fn read_vectored(&self, _data: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        //ManuallyDrop::new(unsafe { WasiFd::from_raw(libc::STDIN_FILENO as u32) })
        //    .read(data)
        Ok(0)
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        true
    }
}

impl Stdout {
    pub fn new() -> io::Result<Stdout> {
        Ok(Stdout)
    }

    pub fn write(&self, data: &[u8]) -> io::Result<usize> {
        let len;

        unsafe { len = abi::write(1, data.as_ptr() as *const u8, data.len()) }

        if len < 0 {
            Err(io::Error::new(io::ErrorKind::Other, "Stdout is not able to print"))
        } else {
            Ok(len as usize)
        }
    }

    pub fn write_vectored(&self, data: &[IoSlice<'_>]) -> io::Result<usize> {
        let len;

        unsafe { len = abi::write(1, data.as_ptr() as *const u8, data.len()) }

        if len < 0 {
            Err(io::Error::new(io::ErrorKind::Other, "Stdout is not able to print"))
        } else {
            Ok(len as usize)
        }
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        true
    }

    pub fn flush(&self) -> io::Result<()> {
        Ok(())
    }
}

impl Stderr {
    pub fn new() -> io::Result<Stderr> {
        Ok(Stderr)
    }

    pub fn write(&self, data: &[u8]) -> io::Result<usize> {
        let len;

        unsafe { len = abi::write(2, data.as_ptr() as *const u8, data.len()) }

        if len < 0 {
            Err(io::Error::new(io::ErrorKind::Other, "Stderr is not able to print"))
        } else {
            Ok(len as usize)
        }
    }

    pub fn write_vectored(&self, data: &[IoSlice<'_>]) -> io::Result<usize> {
        let len;

        unsafe { len = abi::write(2, data.as_ptr() as *const u8, data.len()) }

        if len < 0 {
            Err(io::Error::new(io::ErrorKind::Other, "Stderr is not able to print"))
        } else {
            Ok(len as usize)
        }
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        true
    }

    pub fn flush(&self) -> io::Result<()> {
        Ok(())
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

pub const STDIN_BUF_SIZE: usize = 0;

pub fn is_ebadf(_err: &io::Error) -> bool {
    true
}

pub fn panic_output() -> Option<impl io::Write> {
    Stderr::new().ok()
}
