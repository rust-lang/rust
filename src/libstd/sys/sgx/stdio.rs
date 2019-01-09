use fortanix_sgx_abi as abi;

use io;
use sys::fd::FileDesc;

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
    pub fn new() -> io::Result<Stdin> { Ok(Stdin(())) }

    pub fn read(&self, data: &mut [u8]) -> io::Result<usize> {
        with_std_fd(abi::FD_STDIN, |fd| fd.read(data))
    }
}

impl Stdout {
    pub fn new() -> io::Result<Stdout> { Ok(Stdout(())) }

    pub fn write(&self, data: &[u8]) -> io::Result<usize> {
        with_std_fd(abi::FD_STDOUT, |fd| fd.write(data))
    }

    pub fn flush(&self) -> io::Result<()> {
        with_std_fd(abi::FD_STDOUT, |fd| fd.flush())
    }
}

impl Stderr {
    pub fn new() -> io::Result<Stderr> { Ok(Stderr(())) }

    pub fn write(&self, data: &[u8]) -> io::Result<usize> {
        with_std_fd(abi::FD_STDERR, |fd| fd.write(data))
    }

    pub fn flush(&self) -> io::Result<()> {
        with_std_fd(abi::FD_STDERR, |fd| fd.flush())
    }
}

// FIXME: right now this raw stderr handle is used in a few places because
//        std::io::stderr_raw isn't exposed, but once that's exposed this impl
//        should go away
impl io::Write for Stderr {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        Stderr::write(self, data)
    }

    fn flush(&mut self) -> io::Result<()> {
        Stderr::flush(self)
    }
}

pub const STDIN_BUF_SIZE: usize = ::sys_common::io::DEFAULT_BUF_SIZE;

pub fn is_ebadf(err: &io::Error) -> bool {
    // FIXME: Rust normally maps Unix EBADF to `Other`
    err.raw_os_error() == Some(abi::Error::BrokenPipe as _)
}

pub fn panic_output() -> Option<impl io::Write> {
    super::abi::panic::SgxPanicOutput::new()
}
