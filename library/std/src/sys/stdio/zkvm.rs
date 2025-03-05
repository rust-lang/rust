use crate::io::{self, BorrowedCursor};
use crate::sys::pal::abi::{self, fileno};

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
        Ok(unsafe { abi::sys_read(fileno::STDIN, buf.as_mut_ptr(), buf.len()) })
    }

    fn read_buf(&mut self, mut buf: BorrowedCursor<'_>) -> io::Result<()> {
        unsafe {
            let n = abi::sys_read(fileno::STDIN, buf.as_mut().as_mut_ptr().cast(), buf.capacity());
            buf.advance_unchecked(n);
        }
        Ok(())
    }
}

impl Stdout {
    pub const fn new() -> Stdout {
        Stdout
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        unsafe { abi::sys_write(fileno::STDOUT, buf.as_ptr(), buf.len()) }

        Ok(buf.len())
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
        unsafe { abi::sys_write(fileno::STDERR, buf.as_ptr(), buf.len()) }

        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub const STDIN_BUF_SIZE: usize = crate::sys::io::DEFAULT_BUF_SIZE;

pub fn is_ebadf(_err: &io::Error) -> bool {
    true
}

pub fn panic_output() -> Option<impl io::Write> {
    Some(Stderr::new())
}
