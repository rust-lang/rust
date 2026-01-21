#![forbid(unsafe_op_in_unsafe_fn)]

pub fn abort_internal() -> ! {
    unsafe { libc::abort() }
}

#[inline]
#[cfg(target_env = "p1")]
pub(crate) fn err2io(err: wasi::Errno) -> crate::io::Error {
    crate::io::Error::from_raw_os_error(err.raw().into())
}
