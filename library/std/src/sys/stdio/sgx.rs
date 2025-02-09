use fortanix_sgx_abi as abi;

use crate::io;
use crate::sys::fd::FileDesc;

pub struct Stdin(());
pub struct Stdout(());
pub struct Stderr(());

fn with_std_fd<F: FnOnce(&FileDesc) -> R, R>(fd: abi::Fd, f: F) -> R {
    let fd = FileDesc::new(fd);
    let ret = f(&fd);
    fd.into_raw();
    ret
}

impl Stdin {
    pub const fn new() -> Stdin {
        Stdin(())
    }
}

impl io::Read for Stdin {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        with_std_fd(abi::FD_STDIN, |fd| fd.read(buf))
    }
}

impl Stdout {
    pub const fn new() -> Stdout {
        Stdout(())
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        with_std_fd(abi::FD_STDOUT, |fd| fd.write(buf))
    }

    fn flush(&mut self) -> io::Result<()> {
        with_std_fd(abi::FD_STDOUT, |fd| fd.flush())
    }
}

impl Stderr {
    pub const fn new() -> Stderr {
        Stderr(())
    }
}

impl io::Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        with_std_fd(abi::FD_STDERR, |fd| fd.write(buf))
    }

    fn flush(&mut self) -> io::Result<()> {
        with_std_fd(abi::FD_STDERR, |fd| fd.flush())
    }
}

pub const STDIN_BUF_SIZE: usize = crate::sys::io::DEFAULT_BUF_SIZE;

pub fn is_ebadf(err: &io::Error) -> bool {
    // FIXME: Rust normally maps Unix EBADF to `Uncategorized`
    err.raw_os_error() == Some(abi::Error::BrokenPipe as _)
}

pub fn panic_output() -> Option<impl io::Write> {
    crate::sys::pal::abi::panic::SgxPanicOutput::new()
}
