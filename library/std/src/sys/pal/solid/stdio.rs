use super::abi;
use crate::io;

pub struct Stdin;
pub struct Stdout;
pub struct Stderr;
struct PanicOutput;

impl Stdin {
    pub const fn new() -> Stdin {
        Stdin
    }
}

impl io::Read for Stdin {
    fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
        Ok(0)
    }
}

impl Stdout {
    pub const fn new() -> Stdout {
        Stdout
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        unsafe { abi::SOLID_LOG_write(buf.as_ptr(), buf.len()) };
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
        unsafe { abi::SOLID_LOG_write(buf.as_ptr(), buf.len()) };
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl PanicOutput {
    pub const fn new() -> PanicOutput {
        PanicOutput
    }
}

impl io::Write for PanicOutput {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        unsafe { abi::SOLID_LOG_write(buf.as_ptr(), buf.len()) };
        Ok(buf.len())
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
    Some(PanicOutput::new())
}
