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

#![allow(missing_docs, nonstandard_style, unsafe_op_in_unsafe_fn)]

use crate::intrinsics;
use crate::os::raw::c_char;

pub mod alloc;
pub mod args;
#[path = "../unix/cmath.rs"]
pub mod cmath;
pub mod env;
pub mod fd;
pub mod fs;
pub mod futex;
#[path = "../unsupported/io.rs"]
pub mod io;
pub mod memchr;
pub mod net;
pub mod os;
#[path = "../unix/os_str.rs"]
pub mod os_str;
#[path = "../unix/path.rs"]
pub mod path;
#[path = "../unsupported/pipe.rs"]
pub mod pipe;
#[path = "../unsupported/process.rs"]
pub mod process;
pub mod stdio;
pub mod thread;
pub mod thread_local_dtor;
#[path = "../unsupported/thread_local_key.rs"]
pub mod thread_local_key;
pub mod time;

#[path = "../unix/locks"]
pub mod locks {
    mod futex_condvar;
    mod futex_mutex;
    mod futex_rwlock;
    pub(crate) use futex_condvar::Condvar;
    pub(crate) use futex_mutex::Mutex;
    pub(crate) use futex_rwlock::RwLock;
}

use crate::io::ErrorKind;
use crate::os::hermit::abi;

pub fn unsupported<T>() -> crate::io::Result<T> {
    Err(unsupported_err())
}

pub fn unsupported_err() -> crate::io::Error {
    crate::io::const_io_error!(
        crate::io::ErrorKind::Unsupported,
        "operation not supported on HermitCore yet",
    )
}

pub fn abort_internal() -> ! {
    unsafe {
        abi::abort();
    }
}

// FIXME: just a workaround to test the system
pub fn hashmap_random_keys() -> (u64, u64) {
    (1, 2)
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
    let _ = net::init();
    args::init(argc, argv);
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
    use crate::sys::hermit::thread_local_dtor::run_dtors;
    extern "C" {
        fn main(argc: isize, argv: *const *const c_char) -> i32;
    }

    // initialize environment
    os::init_environment(env as *const *const i8);

    let result = main(argc as isize, argv);

    run_dtors();
    abi::exit(result);
}

pub fn decode_error_kind(errno: i32) -> ErrorKind {
    match errno {
        abi::errno::EACCES => ErrorKind::PermissionDenied,
        abi::errno::EADDRINUSE => ErrorKind::AddrInUse,
        abi::errno::EADDRNOTAVAIL => ErrorKind::AddrNotAvailable,
        abi::errno::EAGAIN => ErrorKind::WouldBlock,
        abi::errno::ECONNABORTED => ErrorKind::ConnectionAborted,
        abi::errno::ECONNREFUSED => ErrorKind::ConnectionRefused,
        abi::errno::ECONNRESET => ErrorKind::ConnectionReset,
        abi::errno::EEXIST => ErrorKind::AlreadyExists,
        abi::errno::EINTR => ErrorKind::Interrupted,
        abi::errno::EINVAL => ErrorKind::InvalidInput,
        abi::errno::ENOENT => ErrorKind::NotFound,
        abi::errno::ENOTCONN => ErrorKind::NotConnected,
        abi::errno::EPERM => ErrorKind::PermissionDenied,
        abi::errno::EPIPE => ErrorKind::BrokenPipe,
        abi::errno::ETIMEDOUT => ErrorKind::TimedOut,
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
            Err(ref e) if e.kind() == ErrorKind::Interrupted => {}
            other => return other,
        }
    }
}
