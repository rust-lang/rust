use fortanix_sgx_abi as abi;

use crate::io;
use crate::sys::fd::FileDesc;
#[cfg(not(test))]
use crate::slice;
#[cfg(not(test))]
use crate::str;

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
}

impl io::Read for Stdin {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        with_std_fd(abi::FD_STDIN, |fd| fd.read(buf))
    }
}

impl Stdout {
    pub fn new() -> io::Result<Stdout> { Ok(Stdout(())) }
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
    pub fn new() -> io::Result<Stderr> { Ok(Stderr(())) }
}

impl io::Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        with_std_fd(abi::FD_STDERR, |fd| fd.write(buf))
    }

    fn flush(&mut self) -> io::Result<()> {
        with_std_fd(abi::FD_STDERR, |fd| fd.flush())
    }
}

pub const STDIN_BUF_SIZE: usize = crate::sys_common::io::DEFAULT_BUF_SIZE;

pub fn is_ebadf(err: &io::Error) -> bool {
    // FIXME: Rust normally maps Unix EBADF to `Other`
    err.raw_os_error() == Some(abi::Error::BrokenPipe as _)
}

pub fn panic_output() -> Option<impl io::Write> {
    super::abi::panic::SgxPanicOutput::new()
}

// This function is needed by libunwind. The symbol is named in pre-link args
// for the target specification, so keep that in sync.
#[cfg(not(test))]
#[no_mangle]
pub unsafe extern "C" fn __rust_print_err(m: *mut u8, s: i32) {
    if s < 0 {
        return;
    }
    let buf = slice::from_raw_parts(m as *const u8, s as _);
    if let Ok(s) = str::from_utf8(&buf[..buf.iter().position(|&b| b == 0).unwrap_or(buf.len())]) {
        eprint!("{}", s);
    }
}
