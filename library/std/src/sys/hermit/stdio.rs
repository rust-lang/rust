use crate::io;
use crate::io::{IoSlice, IoSliceMut};
use crate::sys::hermit::abi;

pub struct Stdin;
pub struct Stdout;
pub struct Stderr;

impl Stdin {
    pub const fn new() -> Stdin {
        Stdin
    }
}

impl io::Read for Stdin {
    fn read(&mut self, data: &mut [u8]) -> io::Result<usize> {
        self.read_vectored(&mut [IoSliceMut::new(data)])
    }

    fn read_vectored(&mut self, _data: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        Ok(0)
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
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        let len;

        unsafe { len = abi::write(1, data.as_ptr() as *const u8, data.len()) }

        if len < 0 {
            Err(io::Error::new_const(io::ErrorKind::Uncategorized, &"Stdout is not able to print"))
        } else {
            Ok(len as usize)
        }
    }

    fn write_vectored(&mut self, data: &[IoSlice<'_>]) -> io::Result<usize> {
        let len;

        unsafe { len = abi::write(1, data.as_ptr() as *const u8, data.len()) }

        if len < 0 {
            Err(io::Error::new_const(io::ErrorKind::Uncategorized, &"Stdout is not able to print"))
        } else {
            Ok(len as usize)
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
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        let len;

        unsafe { len = abi::write(2, data.as_ptr() as *const u8, data.len()) }

        if len < 0 {
            Err(io::Error::new_const(io::ErrorKind::Uncategorized, &"Stderr is not able to print"))
        } else {
            Ok(len as usize)
        }
    }

    fn write_vectored(&mut self, data: &[IoSlice<'_>]) -> io::Result<usize> {
        let len;

        unsafe { len = abi::write(2, data.as_ptr() as *const u8, data.len()) }

        if len < 0 {
            Err(io::Error::new_const(io::ErrorKind::Uncategorized, &"Stderr is not able to print"))
        } else {
            Ok(len as usize)
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

pub const STDIN_BUF_SIZE: usize = 0;

pub fn is_ebadf(_err: &io::Error) -> bool {
    true
}

pub fn panic_output() -> Option<impl io::Write> {
    Some(Stderr::new())
}
