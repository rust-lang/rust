//! UEFI-specific extensions to the primitives in `std::env` module

#![unstable(feature = "uefi_std", issue = "100499")]

use crate::ffi::c_void;
use crate::ptr::NonNull;
use crate::sync::atomic::{Atomic, AtomicBool, AtomicPtr, Ordering};

static SYSTEM_TABLE: Atomic<*mut c_void> = AtomicPtr::new(crate::ptr::null_mut());
static IMAGE_HANDLE: Atomic<*mut c_void> = AtomicPtr::new(crate::ptr::null_mut());
// Flag to check if BootServices are still valid.
// Start with assuming that they are not available
static BOOT_SERVICES_FLAG: Atomic<bool> = AtomicBool::new(false);

/// Initializes the global System Table and Image Handle pointers.
///
/// The standard library requires access to the UEFI System Table and the Application Image Handle
/// to operate. Those are provided to UEFI Applications via their application entry point. By
/// calling `init_globals()`, those pointers are retained by the standard library for future use.
/// Thus this function must be called before any of the standard library services are used.
///
/// The pointers are never exposed to any entity outside of this application and it is guaranteed
/// that, once the application exited, these pointers are never dereferenced again.
///
/// Callers are required to ensure the pointers are valid for the entire lifetime of this
/// application. In particular, UEFI Boot Services must not be exited while an application with the
/// standard library is loaded.
///
/// # SAFETY
/// Calling this function more than once will panic.
pub(crate) unsafe fn init_globals(handle: NonNull<c_void>, system_table: NonNull<c_void>) {
    IMAGE_HANDLE
        .compare_exchange(
            crate::ptr::null_mut(),
            handle.as_ptr(),
            Ordering::Release,
            Ordering::Acquire,
        )
        .unwrap();
    SYSTEM_TABLE
        .compare_exchange(
            crate::ptr::null_mut(),
            system_table.as_ptr(),
            Ordering::Release,
            Ordering::Acquire,
        )
        .unwrap();
    BOOT_SERVICES_FLAG.store(true, Ordering::Release)
}

/// Gets the SystemTable Pointer.
///
/// If you want to use `BootServices` then please use [`boot_services`] as it performs some
/// additional checks.
///
/// Note: This function panics if the System Table or Image Handle is not initialized.
pub fn system_table() -> NonNull<c_void> {
    try_system_table().unwrap()
}

/// Gets the ImageHandle Pointer.
///
/// Note: This function panics if the System Table or Image Handle is not initialized.
pub fn image_handle() -> NonNull<c_void> {
    try_image_handle().unwrap()
}

/// Gets the BootServices Pointer.
///
/// This function also checks if `ExitBootServices` has already been called.
pub fn boot_services() -> Option<NonNull<c_void>> {
    if BOOT_SERVICES_FLAG.load(Ordering::Acquire) {
        let system_table: NonNull<r_efi::efi::SystemTable> = try_system_table()?.cast();
        let boot_services = unsafe { (*system_table.as_ptr()).boot_services };
        NonNull::new(boot_services).map(|x| x.cast())
    } else {
        None
    }
}

/// Gets the SystemTable Pointer.
///
/// This function is mostly intended for places where panic is not an option.
pub(crate) fn try_system_table() -> Option<NonNull<c_void>> {
    NonNull::new(SYSTEM_TABLE.load(Ordering::Acquire))
}

/// Gets the SystemHandle Pointer.
///
/// This function is mostly intended for places where panicking is not an option.
pub(crate) fn try_image_handle() -> Option<NonNull<c_void>> {
    NonNull::new(IMAGE_HANDLE.load(Ordering::Acquire))
}

pub(crate) fn disable_boot_services() {
    BOOT_SERVICES_FLAG.store(false, Ordering::Release)
}
