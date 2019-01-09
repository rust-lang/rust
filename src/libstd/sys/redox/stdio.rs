use io;
use sys::{cvt, syscall};
use sys::fd::FileDesc;

pub struct Stdin(());
pub struct Stdout(());
pub struct Stderr(());

impl Stdin {
    pub fn new() -> io::Result<Stdin> { Ok(Stdin(())) }

    pub fn read(&self, data: &mut [u8]) -> io::Result<usize> {
        let fd = FileDesc::new(0);
        let ret = fd.read(data);
        fd.into_raw();
        ret
    }
}

impl Stdout {
    pub fn new() -> io::Result<Stdout> { Ok(Stdout(())) }

    pub fn write(&self, data: &[u8]) -> io::Result<usize> {
        let fd = FileDesc::new(1);
        let ret = fd.write(data);
        fd.into_raw();
        ret
    }

    pub fn flush(&self) -> io::Result<()> {
        cvt(syscall::fsync(1)).and(Ok(()))
    }
}

impl Stderr {
    pub fn new() -> io::Result<Stderr> { Ok(Stderr(())) }

    pub fn write(&self, data: &[u8]) -> io::Result<usize> {
        let fd = FileDesc::new(2);
        let ret = fd.write(data);
        fd.into_raw();
        ret
    }

    pub fn flush(&self) -> io::Result<()> {
        cvt(syscall::fsync(2)).and(Ok(()))
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

pub fn is_ebadf(err: &io::Error) -> bool {
    err.raw_os_error() == Some(::sys::syscall::EBADF as i32)
}

pub const STDIN_BUF_SIZE: usize = ::sys_common::io::DEFAULT_BUF_SIZE;

pub fn panic_output() -> Option<impl io::Write> {
    Stderr::new().ok()
}
