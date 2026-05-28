//! System bindings for the WASI platforms.
//!
//! This module contains the facade (aka platform-specific) implementations of
//! OS level functionality for WASI. Currently this includes both WASIp1 and
//! WASIp2.

use crate::io;

pub mod conf;
#[allow(unused)]
#[path = "../wasm/atomics/futex.rs"]
pub mod futex;
pub mod stack_overflow;
#[path = "../unix/time.rs"]
pub mod time;

#[cfg(not(target_env = "p1"))]
mod cabi_realloc;

#[path = "../unsupported/common.rs"]
#[deny(unsafe_op_in_unsafe_fn)]
#[expect(dead_code)]
mod common;
pub use common::{cleanup, init, unsupported};

pub fn abort_internal() -> ! {
    unsafe { libc::abort() }
}

#[inline]
#[cfg(target_env = "p1")]
pub(crate) fn err2io(err: wasip1::Errno) -> crate::io::Error {
    crate::io::Error::from_raw_os_error(err.raw().into())
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
            Err(ref e) if e.is_interrupted() => {}
            other => return other,
        }
    }
}
