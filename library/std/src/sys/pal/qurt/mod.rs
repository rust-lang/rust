#![allow(missing_docs, nonstandard_style)]

use crate::io;

#[path = "../unix/conf.rs"]
pub mod conf;
#[path = "../unix/stack_overflow.rs"]
pub mod stack_overflow;
#[path = "../unix/sync/mod.rs"]
pub mod sync;
#[path = "../unix/thread_parking.rs"]
pub mod thread_parking;
#[path = "../unix/time.rs"]
pub mod time;

// SAFETY: must be called only once during runtime initialization.
// NOTE: this is not guaranteed to run, for example when Rust code is called externally.
pub unsafe fn init(argc: isize, argv: *const *const u8, _sigpipe: u8) {
    stack_overflow::init();
    unsafe { crate::sys::args::init(argc, argv) };
}

// SAFETY: must be called only once during runtime cleanup.
// NOTE: this is not guaranteed to run, for example when the program aborts.
pub unsafe fn cleanup() {}

pub fn abort_internal() -> ! {
    unsafe { libc::abort() }
}

pub fn unsupported<T>() -> io::Result<T> {
    Err(unsupported_err())
}

pub fn unsupported_err() -> io::Error {
    io::Error::UNSUPPORTED_PLATFORM
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

/// Converts native return values to Result using the *-1 means error is in `errno`*  convention.
/// Non-error values are `Ok`-wrapped.
pub fn cvt<T: IsMinusOne>(t: T) -> io::Result<T> {
    if t.is_minus_one() { Err(io::Error::last_os_error()) } else { Ok(t) }
}

/// `-1` → look at `errno` → retry on `EINTR`. Otherwise `Ok()`-wrap the closure return value.
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

#[allow(dead_code)] // Not used on all platforms.
/// Zero means `Ok()`, all other values are treated as raw OS errors. Does not look at `errno`.
pub fn cvt_nz(error: libc::c_int) -> io::Result<()> {
    if error == 0 { Ok(()) } else { Err(io::Error::from_raw_os_error(error)) }
}
