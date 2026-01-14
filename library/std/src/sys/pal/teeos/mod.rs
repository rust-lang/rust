//! System bindings for the Teeos platform
//!
//! This module contains the facade (aka platform-specific) implementations of
//! OS level functionality for Teeos.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(unused_variables)]
#![allow(dead_code)]

pub mod os;
#[allow(non_upper_case_globals)]
#[path = "../unix/time.rs"]
pub mod time;

#[path = "../unix/sync"]
pub mod sync {
    mod condvar;
    mod mutex;
    pub use condvar::Condvar;
    pub use mutex::Mutex;
}

use crate::io;

pub fn abort_internal() -> ! {
    unsafe { libc::abort() }
}

// Trusted Applications are loaded as dynamic libraries on Teeos,
// so this should never be called.
pub fn init(argc: isize, argv: *const *const u8, sigpipe: u8) {}

// SAFETY: must be called only once during runtime cleanup.
// this is not guaranteed to run, for example when the program aborts.
pub unsafe fn cleanup() {
    unimplemented!()
    // We do NOT have stack overflow handler, because TEE OS will kill TA when it happens.
    // So cleanup is commented
    // stack_overflow::cleanup();
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

pub fn cvt<T: IsMinusOne>(t: T) -> io::Result<T> {
    if t.is_minus_one() { Err(io::Error::last_os_error()) } else { Ok(t) }
}

pub fn cvt_r<T, F>(mut f: F) -> io::Result<T>
where
    T: IsMinusOne,
    F: FnMut() -> T,
{
    loop {
        match cvt(f()) {
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
            other => return other,
        }
    }
}

pub fn cvt_nz(error: libc::c_int) -> io::Result<()> {
    if error == 0 { Ok(()) } else { Err(io::Error::from_raw_os_error(error)) }
}

pub fn unsupported<T>() -> io::Result<T> {
    Err(unsupported_err())
}

pub fn unsupported_err() -> io::Error {
    io::Error::UNSUPPORTED_PLATFORM
}
