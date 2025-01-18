//! System bindings for HermitCore
//!
//! This module contains the facade (aka platform-specific) implementations of
//! OS level functionality for HermitCore.
//!
//! This is all super highly experimental and not actually intended for
//! wide/production use yet, it's still all in the experimental category. This
//! will likely change over time.
//!
//! Currently all functions here are basically stubs that immediately return
//! errors. The hope is that with a portability lint we can turn actually just
//! remove all this and just omit parts of the standard library if we're
//! compiling for wasm. That way it's a compile time error for something that's
//! guaranteed to be a runtime error!

#![deny(unsafe_op_in_unsafe_fn)]
#![allow(missing_docs, nonstandard_style)]

use crate::os::raw::c_char;

pub mod args;
pub mod env;
pub mod fd;
pub mod fs;
pub mod futex;
pub mod os;
#[path = "../unsupported/pipe.rs"]
pub mod pipe;
#[path = "../unsupported/process.rs"]
pub mod process;
pub mod stdio;
pub mod thread;
pub mod time;

use crate::io::ErrorKind;
use crate::os::hermit::hermit_abi;

pub fn unsupported<T>() -> crate::io::Result<T> {
    Err(unsupported_err())
}

pub fn unsupported_err() -> crate::io::Error {
    crate::io::const_error!(
        crate::io::ErrorKind::Unsupported,
        "operation not supported on HermitCore yet",
    )
}

pub fn abort_internal() -> ! {
    unsafe { hermit_abi::abort() }
}

// This function is needed by the panic runtime. The symbol is named in
// pre-link args for the target specification, so keep that in sync.
#[cfg(not(test))]
#[no_mangle]
// NB. used by both libunwind and libpanic_abort
pub extern "C" fn __rust_abort() {
    abort_internal();
}

// SAFETY: must be called only once during runtime initialization.
// NOTE: this is not guaranteed to run, for example when Rust code is called externally.
pub unsafe fn init(argc: isize, argv: *const *const u8, _sigpipe: u8) {
    unsafe {
        args::init(argc, argv);
    }
}

// SAFETY: must be called only once during runtime cleanup.
// NOTE: this is not guaranteed to run, for example when the program aborts.
pub unsafe fn cleanup() {}

#[cfg(not(test))]
#[no_mangle]
pub unsafe extern "C" fn runtime_entry(
    argc: i32,
    argv: *const *const c_char,
    env: *const *const c_char,
) -> ! {
    extern "C" {
        fn main(argc: isize, argv: *const *const c_char) -> i32;
    }

    // initialize environment
    os::init_environment(env);

    let result = unsafe { main(argc as isize, argv) };

    unsafe {
        crate::sys::thread_local::destructors::run();
    }
    crate::rt::thread_cleanup();

    unsafe {
        hermit_abi::exit(result);
    }
}

#[inline]
pub(crate) fn is_interrupted(errno: i32) -> bool {
    errno == hermit_abi::errno::EINTR
}

pub fn decode_error_kind(errno: i32) -> ErrorKind {
    match errno {
        hermit_abi::errno::EACCES => ErrorKind::PermissionDenied,
        hermit_abi::errno::EADDRINUSE => ErrorKind::AddrInUse,
        hermit_abi::errno::EADDRNOTAVAIL => ErrorKind::AddrNotAvailable,
        hermit_abi::errno::EAGAIN => ErrorKind::WouldBlock,
        hermit_abi::errno::ECONNABORTED => ErrorKind::ConnectionAborted,
        hermit_abi::errno::ECONNREFUSED => ErrorKind::ConnectionRefused,
        hermit_abi::errno::ECONNRESET => ErrorKind::ConnectionReset,
        hermit_abi::errno::EEXIST => ErrorKind::AlreadyExists,
        hermit_abi::errno::EINTR => ErrorKind::Interrupted,
        hermit_abi::errno::EINVAL => ErrorKind::InvalidInput,
        hermit_abi::errno::ENOENT => ErrorKind::NotFound,
        hermit_abi::errno::ENOTCONN => ErrorKind::NotConnected,
        hermit_abi::errno::EPERM => ErrorKind::PermissionDenied,
        hermit_abi::errno::EPIPE => ErrorKind::BrokenPipe,
        hermit_abi::errno::ETIMEDOUT => ErrorKind::TimedOut,
        _ => ErrorKind::Uncategorized,
    }
}

#[doc(hidden)]
pub trait IsNegative {
    fn is_negative(&self) -> bool;
    fn negate(&self) -> i32;
}

macro_rules! impl_is_negative {
    ($($t:ident)*) => ($(impl IsNegative for $t {
        fn is_negative(&self) -> bool {
            *self < 0
        }

        fn negate(&self) -> i32 {
            i32::try_from(-(*self)).unwrap()
        }
    })*)
}

impl IsNegative for i32 {
    fn is_negative(&self) -> bool {
        *self < 0
    }

    fn negate(&self) -> i32 {
        -(*self)
    }
}
impl_is_negative! { i8 i16 i64 isize }

pub fn cvt<T: IsNegative>(t: T) -> crate::io::Result<T> {
    if t.is_negative() {
        let e = decode_error_kind(t.negate());
        Err(crate::io::Error::from(e))
    } else {
        Ok(t)
    }
}

pub fn cvt_r<T, F>(mut f: F) -> crate::io::Result<T>
where
    T: IsNegative,
    F: FnMut() -> T,
{
    loop {
        match cvt(f()) {
            Err(ref e) if e.is_interrupted() => {}
            other => return other,
        }
    }
}
