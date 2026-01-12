#![forbid(unsafe_op_in_unsafe_fn)]

use crate::io as std_io;

pub fn abort_internal() -> ! {
    unsafe { libc::abort() }
}

#[inline]
#[cfg(target_env = "p1")]
pub(crate) fn err2io(err: wasi::Errno) -> std_io::Error {
    std_io::Error::from_raw_os_error(err.raw().into())
}
