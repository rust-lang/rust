use crate::io;
use crate::sys::fd::FileDesc;

pub type Pipe = FileDesc;

#[inline]
pub fn pipe() -> io::Result<(Pipe, Pipe)> {
    Err(io::Error::UNSUPPORTED_PLATFORM)
}
