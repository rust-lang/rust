#![allow(dead_code)]
#![allow(missing_docs, nonstandard_style)]
#![deny(unsafe_op_in_unsafe_fn)]

mod abi;

#[path = "../itron"]
mod itron {
    pub(super) mod abi;
    pub mod condvar;
    pub(super) mod error;
    pub mod mutex;
    pub(super) mod spin;
    pub(super) mod task;
    pub mod thread;
    pub(super) mod time;
    use super::unsupported;
}

pub mod alloc;
#[path = "../unsupported/args.rs"]
pub mod args;
#[path = "../unix/cmath.rs"]
pub mod cmath;
pub mod env;
// `error` is `pub(crate)` so that it can be accessed by `itron/error.rs` as
// `crate::sys::error`
pub(crate) mod error;
pub mod fs;
pub mod io;
pub mod net;
pub mod os;
#[path = "../unix/os_str.rs"]
pub mod os_str;
pub mod path;
#[path = "../unsupported/pipe.rs"]
pub mod pipe;
#[path = "../unsupported/process.rs"]
pub mod process;
pub mod rwlock;
pub mod stdio;
pub use self::itron::{condvar, mutex, thread};
pub mod memchr;
pub mod thread_local_dtor;
pub mod thread_local_key;
pub mod time;

// SAFETY: must be called only once during runtime initialization.
// NOTE: this is not guaranteed to run, for example when Rust code is called externally.
pub unsafe fn init(_argc: isize, _argv: *const *const u8) {}

// SAFETY: must be called only once during runtime cleanup.
pub unsafe fn cleanup() {}

pub fn unsupported<T>() -> crate::io::Result<T> {
    Err(unsupported_err())
}

pub fn unsupported_err() -> crate::io::Error {
    crate::io::Error::new_const(
        crate::io::ErrorKind::Unsupported,
        &"operation not supported on this platform",
    )
}

pub fn decode_error_kind(code: i32) -> crate::io::ErrorKind {
    error::decode_error_kind(code)
}

#[inline(always)]
pub fn abort_internal() -> ! {
    loop {
        abi::breakpoint_abort();
    }
}

// This function is needed by the panic runtime. The symbol is named in
// pre-link args for the target specification, so keep that in sync.
#[cfg(not(test))]
#[no_mangle]
// NB. used by both libunwind and libpanic_abort
pub extern "C" fn __rust_abort() {
    abort_internal();
}

pub fn hashmap_random_keys() -> (u64, u64) {
    unsafe {
        let mut out = crate::mem::MaybeUninit::<[u64; 2]>::uninit();
        let result = abi::SOLID_RNG_SampleRandomBytes(out.as_mut_ptr() as *mut u8, 16);
        assert_eq!(result, 0, "SOLID_RNG_SampleRandomBytes failed: {}", result);
        let [x1, x2] = out.assume_init();
        (x1, x2)
    }
}

pub use libc::strlen;
