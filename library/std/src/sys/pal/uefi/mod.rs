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
#![forbid(unsafe_op_in_unsafe_fn)]

pub mod helpers;
pub mod os;
pub mod time;

#[cfg(test)]
mod tests;

use crate::io;
use crate::os::uefi;
use crate::ptr::NonNull;
use crate::sync::atomic::{Atomic, AtomicPtr, Ordering};

static EXIT_BOOT_SERVICE_EVENT: Atomic<*mut crate::ffi::c_void> =
    AtomicPtr::new(crate::ptr::null_mut());

/// # SAFETY
/// - must be called only once during runtime initialization.
/// - argc must be 2.
/// - argv must be &[Handle, *mut SystemTable].
pub(crate) unsafe fn init(argc: isize, argv: *const *const u8, _sigpipe: u8) {
    assert_eq!(argc, 2);
    let image_handle = unsafe { NonNull::new(*argv as *mut crate::ffi::c_void).unwrap() };
    let system_table = unsafe { NonNull::new(*argv.add(1) as *mut crate::ffi::c_void).unwrap() };
    unsafe { uefi::env::init_globals(image_handle, system_table) };

    // Register exit boot services handler
    match helpers::OwnedEvent::new(
        r_efi::efi::EVT_SIGNAL_EXIT_BOOT_SERVICES,
        r_efi::efi::TPL_NOTIFY,
        Some(exit_boot_service_handler),
        None,
    ) {
        Ok(x) => {
            if EXIT_BOOT_SERVICE_EVENT
                .compare_exchange(
                    crate::ptr::null_mut(),
                    x.into_raw(),
                    Ordering::Release,
                    Ordering::Acquire,
                )
                .is_err()
            {
                abort_internal();
            };
        }
        Err(_) => abort_internal(),
    }
}

/// # SAFETY
/// this is not guaranteed to run, for example when the program aborts.
/// - must be called only once during runtime cleanup.
pub unsafe fn cleanup() {
    if let Some(exit_boot_service_event) =
        NonNull::new(EXIT_BOOT_SERVICE_EVENT.swap(crate::ptr::null_mut(), Ordering::Acquire))
    {
        let _ = unsafe { helpers::OwnedEvent::from_raw(exit_boot_service_event.as_ptr()) };
    }
}

#[inline]
pub const fn unsupported<T>() -> io::Result<T> {
    Err(unsupported_err())
}

#[inline]
pub const fn unsupported_err() -> io::Error {
    io::const_error!(io::ErrorKind::Unsupported, "operation not supported on UEFI")
}

pub fn abort_internal() -> ! {
    if let Some(exit_boot_service_event) =
        NonNull::new(EXIT_BOOT_SERVICE_EVENT.load(Ordering::Acquire))
    {
        let _ = unsafe { helpers::OwnedEvent::from_raw(exit_boot_service_event.as_ptr()) };
    }

    if let (Some(boot_services), Some(handle)) =
        (uefi::env::boot_services(), uefi::env::try_image_handle())
    {
        let boot_services: NonNull<r_efi::efi::BootServices> = boot_services.cast();
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

/// Disable access to BootServices if `EVT_SIGNAL_EXIT_BOOT_SERVICES` is signaled
extern "efiapi" fn exit_boot_service_handler(_e: r_efi::efi::Event, _ctx: *mut crate::ffi::c_void) {
    uefi::env::disable_boot_services();
}
