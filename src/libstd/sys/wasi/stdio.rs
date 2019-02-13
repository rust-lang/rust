use crate::io;
use crate::libc;
use crate::sys::cvt;

pub struct Stdin;
pub struct Stdout;
pub struct Stderr;

impl Stdin {
    pub fn new() -> io::Result<Stdin> {
        Ok(Stdin)
    }

    pub fn read(&self, data: &mut [u8]) -> io::Result<usize> {
        let amt = cvt(unsafe {
            libc::read(libc::STDIN_FILENO, data.as_mut_ptr() as *mut _, data.len())
        })?;
        Ok(amt as usize)
    }
}

impl Stdout {
    pub fn new() -> io::Result<Stdout> {
        Ok(Stdout)
    }

    pub fn write(&self, data: &[u8]) -> io::Result<usize> {
        let amt = cvt(unsafe {
            libc::write(libc::STDOUT_FILENO, data.as_ptr() as *const _, data.len())
        })?;
        Ok(amt as usize)
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
        let amt = cvt(unsafe {
            libc::write(libc::STDERR_FILENO, data.as_ptr() as *const _, data.len())
        })?;
        Ok(amt as usize)
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

pub const STDIN_BUF_SIZE: usize = crate::sys_common::io::DEFAULT_BUF_SIZE;

pub fn is_ebadf(err: &io::Error) -> bool {
    err.raw_os_error() == Some(libc::__WASI_EBADF as i32)
}

pub fn panic_output() -> Option<impl io::Write> {
    Stderr::new().ok()
}
