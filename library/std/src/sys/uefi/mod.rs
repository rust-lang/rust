//! Platform-specific extensions to `std` for Unix platforms.
//!
//! Provides access to platform-level information on Unix platforms, and
//! exposes Unix-specific functions that would otherwise be inappropriate as
//! part of the core `std` library.
//!
//! It exposes more ways to deal with platform-specific strings ([`OsStr`],
//! [`OsString`]), allows to set permissions more granularly, extract low-level
//! file descriptors from files and sockets, and has platform-specific helpers
//! for spawning processes.
//!
//! [`OsStr`]: crate::ffi::OsStr
//! [`OsString`]: crate::ffi::OsString

#![deny(unsafe_op_in_unsafe_fn)]

pub mod alloc;
#[path = "../unsupported/args.rs"]
pub mod args;
#[path = "../unix/cmath.rs"]
pub mod cmath;
pub mod env;
#[path = "../unsupported/fs.rs"]
pub mod fs;
#[path = "../unsupported/io.rs"]
pub mod io;
#[path = "../unsupported/locks/mod.rs"]
pub mod locks;
#[path = "../unsupported/net.rs"]
pub mod net;
pub mod os;
pub mod os_str;
#[path = "../unix/path.rs"]
pub mod path;
#[path = "../unsupported/pipe.rs"]
pub mod pipe;
#[path = "../unsupported/process.rs"]
pub mod process;
pub mod stdio;
#[path = "../unsupported/thread.rs"]
pub mod thread;
#[path = "../unsupported/thread_local_key.rs"]
pub mod thread_local_key;
#[path = "../unsupported/time.rs"]
pub mod time;

#[cfg(test)]
mod tests;

use crate::io as std_io;
use crate::os::uefi;
use r_efi::efi;

pub mod memchr {
    pub use core::slice::memchr::{memchr, memrchr};
}

pub unsafe fn init(_argc: isize, _argv: *const *const u8) {}

// SAFETY: must be called only once during runtime cleanup.
// NOTE: this is not guaranteed to run, for example when the program aborts.
pub unsafe fn cleanup() {}

pub fn unsupported<T>() -> std_io::Result<T> {
    Err(unsupported_err())
}

pub fn unsupported_err() -> std_io::Error {
    std_io::const_io_error!(
        std_io::ErrorKind::Unsupported,
        "operation not supported on this platform",
    )
}

pub fn decode_error_kind(_code: i32) -> crate::io::ErrorKind {
    crate::io::ErrorKind::Uncategorized
}

/// FIXME: Check if `exit()` should be used here
pub fn abort_internal() -> ! {
    core::intrinsics::abort();
}

pub fn hashmap_random_keys() -> (u64, u64) {
    (1, 2)
}

extern "C" {
    fn main(argc: isize, argv: *const *const u8) -> isize;
}

#[no_mangle]
pub unsafe extern "efiapi" fn efi_main(
    handle: efi::Handle,
    st: *mut efi::SystemTable,
) -> efi::Status {
    unsafe {
        let mut msg = [
            0x0048u16, 0x0065u16, 0x006cu16, 0x006cu16, 0x006fu16, 0x000du16, 0x000au16, 0x0000u16,
        ];
        ((*(*st).std_err).output_string)((*st).std_err, msg.as_mut_ptr());
    }

    unsafe { uefi::env::init_globals(handle, st).unwrap() };

    match unsafe { main(0, crate::ptr::null()) } {
        0 => efi::Status::SUCCESS,
        _ => efi::Status::ABORTED, // Or some other status code
    }
}
