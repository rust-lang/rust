//! UEFI-specific extensions to the primitives in `std::env` module

#![unstable(feature = "uefi_std", issue = "100499")]

use crate::{cell::Cell, ffi::c_void, ptr::NonNull};

// Since UEFI is single-threaded, making the global variables thread local should be safe.
thread_local! {
    // Flag to check if BootServices are still valid.
    // Start with assuming that they are not available
    static BOOT_SERVICES_FLAG: Cell<bool> = Cell::new(false);
    // Position 0 = SystemTable
    // Position 1 = ImageHandle
    static GLOBALS: Cell<Option<(NonNull<c_void>, NonNull<c_void>)>> = Cell::new(None);
}

/// Initializes the global System Table and Image Handle pointers.
///
/// The standard library requires access to the UEFI System Table and the Application Image Handle
/// to operate. Those are provided to UEFI Applications via their application entry point. By
/// calling `init_globals()`, those pointers are retained by the standard library for future use.
/// The pointers are never exposed to any entity outside of this application and it is guaranteed
/// that, once the application exited, these pointers are never dereferenced again.
///
/// Callers are required to ensure the pointers are valid for the entire lifetime of this
/// application. In particular, UEFI Boot Services must not be exited while an application with the
/// standard library is loaded.
///
/// This function must not be called more than once.
pub unsafe fn init_globals(handle: NonNull<c_void>, system_table: NonNull<c_void>) {
    GLOBALS.set(Some((system_table, handle)));
}

/// Get the SystemTable Pointer.
/// Note: This function panics if the System Table or Image Handle is not initialized
pub fn system_table() -> NonNull<c_void> {
    try_system_table().unwrap()
}

/// Get the ImageHandle Pointer.
/// Note: This function panics if the System Table or Image Handle is not initialized
pub fn image_handle() -> NonNull<c_void> {
    try_image_handle().unwrap()
}

/// Get the SystemTable Pointer.
/// This function is mostly intended for places where panic is not an option
pub(crate) fn try_system_table() -> Option<NonNull<crate::ffi::c_void>> {
    GLOBALS.get().map(|x| x.0)
}

/// Get the SystemHandle Pointer.
/// This function is mostly intended for places where panic is not an option
pub(crate) fn try_image_handle() -> Option<NonNull<crate::ffi::c_void>> {
    GLOBALS.get().map(|x| x.1)
}

/// Get the BootServices Pointer.
/// This function also checks if `ExitBootServices` has already been called.
pub(crate) fn boot_services() -> Option<NonNull<r_efi::efi::BootServices>> {
    if BOOT_SERVICES_FLAG.get() {
        let system_table: NonNull<r_efi::efi::SystemTable> = try_system_table()?.cast();
        let boot_services = unsafe { (*system_table.as_ptr()).boot_services };
        NonNull::new(boot_services)
    } else {
        None
    }
}

pub(crate) fn enable_boot_services() {
    BOOT_SERVICES_FLAG.set(true);
}

pub(crate) fn disable_boot_services() {
    BOOT_SERVICES_FLAG.set(false);
}
