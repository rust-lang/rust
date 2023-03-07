use crate::io::{default_read, BorrowedCursor, Error, Read, Result};
use crate::ptr;
use crate::sys::c;

pub const INSECURE_HASHMAP: bool = false;

pub struct Entropy {
    pub insecure: bool,
}

impl Read for Entropy {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        default_read(self, buf)
    }

    fn read_buf(&mut self, mut buf: BorrowedCursor<'_>) -> Result<()> {
        let len = buf.capacity().try_into().unwrap_or(c::ULONG::MAX);
        let ret = unsafe {
            c::BCryptGenRandom(
                ptr::null_mut(),
                buf.as_ptr().cast(),
                len,
                c::BCRYPT_USE_SYSTEM_PREFERRED_RNG,
            )
        };
        if c::nt_success(ret) {
            unsafe {
                buf.advance(len as usize);
                Ok(())
            }
        } else {
            fallback_rng(buf)
        }
    }
}

/// Generate random numbers using the fallback RNG function (RtlGenRandom)
///
/// This is necessary because of a failure to load the SysWOW64 variant of the
/// bcryptprimitives.dll library from code that lives in bcrypt.dll
/// See <https://bugzilla.mozilla.org/show_bug.cgi?id=1788004#c9>
#[cfg(not(target_vendor = "uwp"))]
#[inline(never)]
fn fallback_rng(mut buf: BorrowedCursor<'_>) -> Result<()> {
    let len = buf.capacity().try_into().unwrap_or(c::ULONG::MAX);
    let ret = unsafe { c::RtlGenRandom(buf.as_ptr().cast(), len) };

    if ret != 0 { Ok(()) } else { Err(Error::last_os_error()) }
}

/// We can't use RtlGenRandom with UWP, so there is no fallback
#[cfg(target_vendor = "uwp")]
#[inline(never)]
fn fallback_rng(_: BorrowedCursor<'_>) -> Result<()> {
    Err(const_io_error!(
        ErrorKind::Unsupported,
        "fallback RNG broken: RtlGenRandom() not supported on UWP"
    ))
}
