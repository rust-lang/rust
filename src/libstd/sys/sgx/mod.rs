//! System bindings for the Fortanix SGX platform
//!
//! This module contains the facade (aka platform-specific) implementations of
//! OS level functionality for Fortanix SGX.

use io;
use os::raw::c_char;
use sync::atomic::{AtomicBool, Ordering};

pub mod abi;
mod waitqueue;

pub mod alloc;
pub mod args;
#[cfg(feature = "backtrace")]
pub mod backtrace;
pub mod cmath;
pub mod condvar;
pub mod env;
pub mod ext;
pub mod fd;
pub mod fs;
pub mod memchr;
pub mod mutex;
pub mod net;
pub mod os;
pub mod os_str;
pub mod path;
pub mod pipe;
pub mod process;
pub mod rwlock;
pub mod stack_overflow;
pub mod thread;
pub mod thread_local;
pub mod time;
pub mod stdio;

#[cfg(not(test))]
pub fn init() {
}

/// This function is used to implement functionality that simply doesn't exist.
/// Programs relying on this functionality will need to deal with the error.
pub fn unsupported<T>() -> io::Result<T> {
    Err(unsupported_err())
}

pub fn unsupported_err() -> io::Error {
    io::Error::new(io::ErrorKind::Other,
                   "operation not supported on SGX yet")
}

/// This function is used to implement various functions that doesn't exist,
/// but the lack of which might not be reason for error. If no error is
/// returned, the program might very well be able to function normally. This is
/// what happens when `SGX_INEFFECTIVE_ERROR` is set to `true`. If it is
/// `false`, the behavior is the same as `unsupported`.
pub fn sgx_ineffective<T>(v: T) -> io::Result<T> {
    static SGX_INEFFECTIVE_ERROR: AtomicBool = AtomicBool::new(false);
    if SGX_INEFFECTIVE_ERROR.load(Ordering::Relaxed) {
        Err(io::Error::new(io::ErrorKind::Other,
                       "operation can't be trusted to have any effect on SGX"))
    } else {
        Ok(v)
    }
}

pub fn decode_error_kind(code: i32) -> io::ErrorKind {
    use fortanix_sgx_abi::Error;

    // FIXME: not sure how to make sure all variants of Error are covered
    if code == Error::NotFound as _ {
        io::ErrorKind::NotFound
    } else if code == Error::PermissionDenied as _ {
        io::ErrorKind::PermissionDenied
    } else if code == Error::ConnectionRefused as _ {
        io::ErrorKind::ConnectionRefused
    } else if code == Error::ConnectionReset as _ {
        io::ErrorKind::ConnectionReset
    } else if code == Error::ConnectionAborted as _ {
        io::ErrorKind::ConnectionAborted
    } else if code == Error::NotConnected as _ {
        io::ErrorKind::NotConnected
    } else if code == Error::AddrInUse as _ {
        io::ErrorKind::AddrInUse
    } else if code == Error::AddrNotAvailable as _ {
        io::ErrorKind::AddrNotAvailable
    } else if code == Error::BrokenPipe as _ {
        io::ErrorKind::BrokenPipe
    } else if code == Error::AlreadyExists as _ {
        io::ErrorKind::AlreadyExists
    } else if code == Error::WouldBlock as _ {
        io::ErrorKind::WouldBlock
    } else if code == Error::InvalidInput as _ {
        io::ErrorKind::InvalidInput
    } else if code == Error::InvalidData as _ {
        io::ErrorKind::InvalidData
    } else if code == Error::TimedOut as _ {
        io::ErrorKind::TimedOut
    } else if code == Error::WriteZero as _ {
        io::ErrorKind::WriteZero
    } else if code == Error::Interrupted as _ {
        io::ErrorKind::Interrupted
    } else if code == Error::Other as _ {
        io::ErrorKind::Other
    } else if code == Error::UnexpectedEof as _ {
        io::ErrorKind::UnexpectedEof
    } else {
        io::ErrorKind::Other
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
    return n
}

pub unsafe fn abort_internal() -> ! {
    abi::panic::panic_exit()
}

pub fn hashmap_random_keys() -> (u64, u64) {
    fn rdrand64() -> u64 {
        unsafe {
            let mut ret: u64 = ::mem::uninitialized();
            for _ in 0..10 {
                if ::arch::x86_64::_rdrand64_step(&mut ret) == 1 {
                    return ret;
                }
            }
            panic!("Failed to obtain random data");
        }
    }
    (rdrand64(), rdrand64())
}

pub use sys_common::{AsInner, FromInner, IntoInner};

pub trait TryIntoInner<Inner>: Sized {
    fn try_into_inner(self) -> Result<Inner, Self>;
}
