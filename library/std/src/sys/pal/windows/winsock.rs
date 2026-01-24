use super::c;
use crate::ffi::c_int;
use crate::sync::atomic::Atomic;
use crate::sync::atomic::Ordering::{AcqRel, Relaxed};
use crate::{io, mem};

static WSA_STARTED: Atomic<bool> = Atomic::<bool>::new(false);

/// Checks whether the Windows socket interface has been started already, and
/// if not, starts it.
#[inline]
pub fn startup() {
    if !WSA_STARTED.load(Relaxed) {
        wsa_startup();
    }
}

#[cold]
fn wsa_startup() {
    unsafe {
        let mut data: c::WSADATA = mem::zeroed();
        let ret = c::WSAStartup(
            0x202, // version 2.2
            &mut data,
        );
        assert_eq!(ret, 0);
        if WSA_STARTED.swap(true, AcqRel) {
            // If another thread raced with us and called WSAStartup first then call
            // WSACleanup so it's as though WSAStartup was only called once.
            c::WSACleanup();
        }
    }
}

pub fn cleanup() {
    // We don't need to call WSACleanup here because exiting the process will cause
    // the OS to clean everything for us, which is faster than doing it manually.
    // See #141799.
}

/// Returns the last error from the Windows socket interface.
pub fn last_error() -> io::Error {
    io::Error::from_raw_os_error(unsafe { c::WSAGetLastError() })
}

#[doc(hidden)]
pub trait IsMinusOne {
    fn is_minus_one(&self) -> bool;
}

macro_rules! impl_is_minus_one {
    ($($t:ident)*) => ($(impl IsMinusOne for $t {
        fn is_minus_one(&self) -> bool {
            *self == -1
        }
    })*)
}

impl_is_minus_one! { i8 i16 i32 i64 isize }

/// Checks if the signed integer is the Windows constant `SOCKET_ERROR` (-1)
/// and if so, returns the last error from the Windows socket interface. This
/// function must be called before another call to the socket API is made.
pub fn cvt<T: IsMinusOne>(t: T) -> io::Result<T> {
    if t.is_minus_one() { Err(last_error()) } else { Ok(t) }
}

/// A variant of `cvt` for `getaddrinfo` which return 0 for a success.
pub fn cvt_gai(err: c_int) -> io::Result<()> {
    if err == 0 { Ok(()) } else { Err(last_error()) }
}

/// Just to provide the same interface as sys/pal/unix/net.rs
pub fn cvt_r<T, F>(mut f: F) -> io::Result<T>
where
    T: IsMinusOne,
    F: FnMut() -> T,
{
    cvt(f())
}

#[cfg(target_vendor = "win7")]
pub unsafe fn hostname_fallback(name: c::PWSTR, namelen: i32) -> i32 {
    assert!(namelen >= 1);

    // The documentation of gethostname says that a buffer size of 256 is
    // always enough.
    let mut buffer = [const { mem::MaybeUninit::<u8>::uninit() }; 256];

    // SAFETY: these parameters specify a valid, writable region of memory.
    unsafe {
        if c::gethostname(buffer.as_mut_ptr().cast(), buffer.len() as i32) == c::SOCKET_ERROR {
            return c::SOCKET_ERROR;
        }

        // Subtract one to leave space for the null terminator, as MultiByteToWideChar doesn't terminate the output buffer
        // if the number of output characters is equal to the buffer length.
        let len = c::MultiByteToWideChar(
            c::CP_ACP,
            c::MB_ERR_INVALID_CHARS,
            buffer.as_mut_ptr().cast(),
            -1,
            name,
            namelen - 1,
        );

        if len <= 0 {
            // GetHostNameW reports WSAEFAULT if the buffer is too small
            if c::GetLastError() == c::ERROR_INSUFFICIENT_BUFFER {
                c::SetLastError(c::WSAEFAULT as _);
            }
            return c::SOCKET_ERROR;
        }

        // Ensure the output is always null terminated.
        // If MultiByteToWideChar has already written a null terminator, that null terminator will be included in len
        // and this will add a second one, but writing a zero is cheap enough to omit the length comparison.
        name.add(len as _).write(0);
    }

    // Success
    0
}
