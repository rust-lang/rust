//! System bindings for the Fortanix SGX platform
//!
//! This module contains the facade (aka platform-specific) implementations of
//! OS level functionality for Fortanix SGX.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(fuzzy_provenance_casts)] // FIXME: this entire module systematically confuses pointers and integers

use crate::io;
use crate::sync::atomic::{Atomic, AtomicBool, Ordering};

pub mod abi;
mod libunwind_integration;
pub mod os;
pub mod thread_parking;
pub mod time;
pub mod waitqueue;

// SAFETY: must be called only once during runtime initialization.
// NOTE: this is not guaranteed to run, for example when Rust code is called externally.
pub unsafe fn init(argc: isize, argv: *const *const u8, _sigpipe: u8) {
    unsafe {
        crate::sys::args::init(argc, argv);
    }
}

// SAFETY: must be called only once during runtime cleanup.
// NOTE: this is not guaranteed to run, for example when the program aborts.
pub unsafe fn cleanup() {}

/// This function is used to implement functionality that simply doesn't exist.
/// Programs relying on this functionality will need to deal with the error.
pub fn unsupported<T>() -> io::Result<T> {
    Err(unsupported_err())
}

pub fn unsupported_err() -> io::Error {
    io::const_error!(io::ErrorKind::Unsupported, "operation not supported on SGX yet")
}

/// This function is used to implement various functions that doesn't exist,
/// but the lack of which might not be reason for error. If no error is
/// returned, the program might very well be able to function normally. This is
/// what happens when `SGX_INEFFECTIVE_ERROR` is set to `true`. If it is
/// `false`, the behavior is the same as `unsupported`.
pub fn sgx_ineffective<T>(v: T) -> io::Result<T> {
    static SGX_INEFFECTIVE_ERROR: Atomic<bool> = AtomicBool::new(false);
    if SGX_INEFFECTIVE_ERROR.load(Ordering::Relaxed) {
        Err(io::const_error!(
            io::ErrorKind::Uncategorized,
            "operation can't be trusted to have any effect on SGX",
        ))
    } else {
        Ok(v)
    }
}

#[inline]
pub fn is_interrupted(code: i32) -> bool {
    code == fortanix_sgx_abi::Error::Interrupted as _
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
        io::ErrorKind::Uncategorized
    } else if code == Error::UnexpectedEof as _ {
        io::ErrorKind::UnexpectedEof
    } else {
        io::ErrorKind::Uncategorized
    }
}

pub fn abort_internal() -> ! {
    abi::usercalls::exit(true)
}

// This function is needed by libunwind. The symbol is named in
// pre-link args for the target specification, so keep that in sync.
// Note: contrary to the `__rust_abort` in `crate::rt`, this uses `no_mangle`
//       because it is actually used from C code. Because symbols annotated with
//       #[rustc_std_internal_symbol] get mangled, this will not lead to linker
//       conflicts.
#[cfg(not(test))]
#[unsafe(no_mangle)]
pub extern "C" fn __rust_abort() {
    abort_internal();
}

pub trait TryIntoInner<Inner>: Sized {
    fn try_into_inner(self) -> Result<Inner, Self>;
}
