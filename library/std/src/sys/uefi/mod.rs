//! Platform-specific extensions to `std` for UEFI platforms.
//!
//! Provides access to platform-level information on UEFI platforms, and
//! exposes UEFI-specific functions that would otherwise be inappropriate as
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
#[path = "../unsupported/once.rs"]
pub mod once;
#[path = "../unsupported/os.rs"]
pub mod os;
#[path = "../windows/os_str.rs"]
pub mod os_str;
pub mod path;
#[path = "../unsupported/pipe.rs"]
pub mod pipe;
#[path = "../unsupported/process.rs"]
pub mod process;
#[path = "../unsupported/stdio.rs"]
pub mod stdio;
#[path = "../unsupported/thread.rs"]
pub mod thread;
#[path = "../unsupported/thread_local_key.rs"]
pub mod thread_local_key;
#[path = "../unsupported/time.rs"]
pub mod time;

pub(crate) mod helpers;

#[cfg(test)]
mod tests;

use crate::io as std_io;
use crate::os::uefi;
use crate::ptr::NonNull;

pub mod memchr {
    pub use core::slice::memchr::{memchr, memrchr};
}

/// # SAFETY
/// - must be called only once during runtime initialization.
/// - argc must be 2.
/// - argv must be &[Handle, *mut SystemTable].
pub(crate) unsafe fn init(argc: isize, argv: *const *const u8, _sigpipe: u8) {
    assert_eq!(argc, 2);
    let image_handle = unsafe { NonNull::new(*argv as *mut crate::ffi::c_void).unwrap() };
    let system_table = unsafe { NonNull::new(*argv.add(1) as *mut crate::ffi::c_void).unwrap() };
    unsafe { crate::os::uefi::env::init_globals(image_handle, system_table) };
}

/// # SAFETY
/// this is not guaranteed to run, for example when the program aborts.
/// - must be called only once during runtime cleanup.
pub unsafe fn cleanup() {}

#[inline]
pub const fn unsupported<T>() -> std_io::Result<T> {
    Err(unsupported_err())
}

#[inline]
pub const fn unsupported_err() -> std_io::Error {
    std_io::const_io_error!(std_io::ErrorKind::Unsupported, "operation not supported on UEFI",)
}

pub fn decode_error_kind(code: i32) -> crate::io::ErrorKind {
    use crate::io::ErrorKind;
    use r_efi::efi::Status;

    if let Ok(code) = usize::try_from(code) {
        helpers::status_to_io_error(Status::from_usize(code)).kind()
    } else {
        ErrorKind::Uncategorized
    }
}

pub fn abort_internal() -> ! {
    if let (Some(boot_services), Some(handle)) =
        (helpers::try_boot_services(), uefi::env::try_image_handle())
    {
        let _ = unsafe {
            ((*boot_services.as_ptr()).exit)(
                handle.as_ptr(),
                r_efi::efi::Status::ABORTED,
                0,
                crate::ptr::null_mut(),
            )
        };
    }

    // In case SystemTable and ImageHandle cannot be reached, use `core::intrinsics::abort`
    core::intrinsics::abort();
}

// This function is needed by the panic runtime. The symbol is named in
// pre-link args for the target specification, so keep that in sync.
#[cfg(not(test))]
#[no_mangle]
pub extern "C" fn __rust_abort() {
    abort_internal();
}

#[inline]
pub fn hashmap_random_keys() -> (u64, u64) {
    get_random().unwrap()
}

fn get_random() -> Option<(u64, u64)> {
    use r_efi::protocols::rng;

    let mut buf = [0u8; 16];
    let handles = helpers::locate_handles(rng::PROTOCOL_GUID).ok()?;
    for handle in handles {
        if let Ok(protocol) = helpers::open_protocol::<rng::Protocol>(handle, rng::PROTOCOL_GUID) {
            let r = unsafe {
                ((*protocol.as_ptr()).get_rng)(
                    protocol.as_ptr(),
                    crate::ptr::null_mut(),
                    buf.len(),
                    buf.as_mut_ptr(),
                )
            };
            if r.is_error() {
                continue;
            } else {
                return Some((
                    u64::from_le_bytes(buf[..8].try_into().ok()?),
                    u64::from_le_bytes(buf[8..].try_into().ok()?),
                ));
            }
        }
    }
    None
}
