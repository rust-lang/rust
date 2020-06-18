//! System bindings for the Fortanix SGX platform
//!
//! This module contains the facade (aka platform-specific) implementations of
//! OS level functionality for Fortanix SGX.

use crate::io::ErrorKind;
use crate::os::raw::c_char;
use crate::sync::atomic::{AtomicBool, Ordering};

pub mod abi;
mod waitqueue;

pub mod alloc;
pub mod args;
pub mod cmath;
pub mod condvar;
pub mod env;
pub mod ext;
pub mod fd;
pub mod fs;
pub mod io;
pub mod memchr;
pub mod mutex;
pub mod net;
pub mod os;
pub mod path;
pub mod pipe;
pub mod process;
pub mod rwlock;
pub mod stack_overflow;
pub mod stdio;
pub mod thread;
pub mod thread_local;
pub mod time;

pub use crate::sys_common::os_str_bytes as os_str;

#[cfg(not(test))]
pub fn init() {}

/// This function is used to implement functionality that simply doesn't exist.
/// Programs relying on this functionality will need to deal with the error.
pub fn unsupported<T>() -> crate::io::Result<T> {
    Err(unsupported_err())
}

pub fn unsupported_err() -> crate::io::Error {
    crate::io::Error::new(ErrorKind::Other, "operation not supported on SGX yet")
}

/// This function is used to implement various functions that doesn't exist,
/// but the lack of which might not be reason for error. If no error is
/// returned, the program might very well be able to function normally. This is
/// what happens when `SGX_INEFFECTIVE_ERROR` is set to `true`. If it is
/// `false`, the behavior is the same as `unsupported`.
pub fn sgx_ineffective<T>(v: T) -> crate::io::Result<T> {
    static SGX_INEFFECTIVE_ERROR: AtomicBool = AtomicBool::new(false);
    if SGX_INEFFECTIVE_ERROR.load(Ordering::Relaxed) {
        Err(crate::io::Error::new(
            ErrorKind::Other,
            "operation can't be trusted to have any effect on SGX",
        ))
    } else {
        Ok(v)
    }
}

pub fn decode_error_kind(code: i32) -> ErrorKind {
    use fortanix_sgx_abi::Error;

    // FIXME: not sure how to make sure all variants of Error are covered
    if code == Error::NotFound as _ {
        ErrorKind::NotFound
    } else if code == Error::PermissionDenied as _ {
        ErrorKind::PermissionDenied
    } else if code == Error::ConnectionRefused as _ {
        ErrorKind::ConnectionRefused
    } else if code == Error::ConnectionReset as _ {
        ErrorKind::ConnectionReset
    } else if code == Error::ConnectionAborted as _ {
        ErrorKind::ConnectionAborted
    } else if code == Error::NotConnected as _ {
        ErrorKind::NotConnected
    } else if code == Error::AddrInUse as _ {
        ErrorKind::AddrInUse
    } else if code == Error::AddrNotAvailable as _ {
        ErrorKind::AddrNotAvailable
    } else if code == Error::BrokenPipe as _ {
        ErrorKind::BrokenPipe
    } else if code == Error::AlreadyExists as _ {
        ErrorKind::AlreadyExists
    } else if code == Error::WouldBlock as _ {
        ErrorKind::WouldBlock
    } else if code == Error::InvalidInput as _ {
        ErrorKind::InvalidInput
    } else if code == Error::InvalidData as _ {
        ErrorKind::InvalidData
    } else if code == Error::TimedOut as _ {
        ErrorKind::TimedOut
    } else if code == Error::WriteZero as _ {
        ErrorKind::WriteZero
    } else if code == Error::Interrupted as _ {
        ErrorKind::Interrupted
    } else if code == Error::Other as _ {
        ErrorKind::Other
    } else if code == Error::UnexpectedEof as _ {
        ErrorKind::UnexpectedEof
    } else {
        ErrorKind::Other
    }
}

// This function makes an effort to wait for a non-spurious event at least as
// long as `duration`. Note that in general there is no guarantee about accuracy
// of time and timeouts in SGX model. The enclave runner serving usercalls may
// lie about current time and/or ignore timeout values.
//
// Once the event is observed, `woken_up` will be used to determine whether or
// not the event was spurious.
//
// FIXME: note these caveats in documentation of all public types that use this
// function in their execution path.
pub fn wait_timeout_sgx<F>(event_mask: u64, duration: crate::time::Duration, woken_up: F)
where
    F: Fn() -> bool,
{
    use self::abi::usercalls;
    use crate::cmp;
    use crate::io::ErrorKind;
    use crate::time::{Duration, Instant};

    // Calls the wait usercall and checks the result. Returns true if event was
    // returned, and false if WouldBlock/TimedOut was returned.
    // If duration is None, it will use WAIT_NO.
    fn wait_checked(event_mask: u64, duration: Option<Duration>) -> bool {
        let timeout = duration.map_or(usercalls::raw::WAIT_NO, |duration| {
            cmp::min((u64::MAX - 1) as u128, duration.as_nanos()) as u64
        });
        match usercalls::wait(event_mask, timeout) {
            Ok(eventset) => {
                if event_mask == 0 {
                    rtabort!("expected usercalls::wait() to return Err, found Ok.");
                }
                rtassert!(eventset & event_mask == event_mask);
                true
            }
            Err(e) => {
                rtassert!(e.kind() == ErrorKind::TimedOut || e.kind() == ErrorKind::WouldBlock);
                false
            }
        }
    }

    match wait_checked(event_mask, Some(duration)) {
        false => return,              // timed out
        true if woken_up() => return, // woken up
        true => {}                    // spurious event
    }

    // Drain all cached events.
    // Note that `event_mask != 0` is implied if we get here.
    loop {
        match wait_checked(event_mask, None) {
            false => break,               // no more cached events
            true if woken_up() => return, // woken up
            true => {}                    // spurious event
        }
    }

    // Continue waiting, but take note of time spent waiting so we don't wait
    // forever. We intentionally don't call `Instant::now()` before this point
    // to avoid the cost of the `insecure_time` usercall in case there are no
    // spurious wakeups.

    let start = Instant::now();
    let mut remaining = duration;
    loop {
        match wait_checked(event_mask, Some(remaining)) {
            false => return,              // timed out
            true if woken_up() => return, // woken up
            true => {}                    // spurious event
        }
        remaining = match duration.checked_sub(start.elapsed()) {
            Some(remaining) => remaining,
            None => break,
        }
    }
}

// This enum is used as the storage for a bunch of types which can't actually
// exist.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Void {}

pub unsafe fn strlen(mut s: *const c_char) -> usize {
    let mut n = 0;
    while *s != 0 {
        n += 1;
        s = s.offset(1);
    }
    return n;
}

pub fn abort_internal() -> ! {
    abi::usercalls::exit(true)
}

// This function is needed by the panic runtime. The symbol is named in
// pre-link args for the target specification, so keep that in sync.
#[cfg(not(test))]
#[no_mangle]
// NB. used by both libunwind and libpanic_abort
pub extern "C" fn __rust_abort() {
    abort_internal();
}

pub mod rand {
    pub fn rdrand64() -> u64 {
        unsafe {
            let mut ret: u64 = 0;
            for _ in 0..10 {
                if crate::arch::x86_64::_rdrand64_step(&mut ret) == 1 {
                    return ret;
                }
            }
            rtabort!("Failed to obtain random data");
        }
    }
}

pub fn hashmap_random_keys() -> (u64, u64) {
    (self::rand::rdrand64(), self::rand::rdrand64())
}

pub use crate::sys_common::{AsInner, FromInner, IntoInner};

pub trait TryIntoInner<Inner>: Sized {
    fn try_into_inner(self) -> Result<Inner, Self>;
}
