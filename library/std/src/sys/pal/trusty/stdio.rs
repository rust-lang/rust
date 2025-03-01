use crate::io;

pub struct Stdin;
pub struct Stdout;
pub struct Stderr;

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
        _write(libc::STDOUT_FILENO, buf)
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
        _write(libc::STDERR_FILENO, buf)
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
    Some(Stderr)
}

fn _write(fd: i32, message: &[u8]) -> io::Result<usize> {
    let mut iov = libc::iovec { iov_base: message.as_ptr() as *mut _, iov_len: message.len() };
    loop {
        // SAFETY: syscall, safe arguments.
        let ret = unsafe { libc::writev(fd, &iov, 1) };
        if ret < 0 {
            return Err(io::Error::last_os_error());
        }
        let ret = ret as usize;
        if ret > iov.iov_len {
            return Err(io::Error::last_os_error());
        }
        if ret == iov.iov_len {
            return Ok(message.len());
        }
        // SAFETY: ret has been checked to be less than the length of
        // the buffer
        iov.iov_base = unsafe { iov.iov_base.add(ret) };
        iov.iov_len -= ret;
    }
}
