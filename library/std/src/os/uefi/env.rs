//! UEFI-specific extensions to the primitives in `std::env` module

#![unstable(feature = "uefi_std", issue = "100499")]

use crate::ffi::c_void;
use crate::ptr::NonNull;
use crate::sync::atomic::{AtomicPtr, Ordering};
use crate::sync::OnceLock;

// Position 0 = SystemTable
// Position 1 = ImageHandle
static GLOBALS: OnceLock<(AtomicPtr<c_void>, AtomicPtr<c_void>)> = OnceLock::new();

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
    GLOBALS.set((AtomicPtr::new(system_table.as_ptr()), AtomicPtr::new(handle.as_ptr()))).unwrap()
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
    NonNull::new(GLOBALS.get()?.0.load(Ordering::Acquire))
}

/// Get the SystemHandle Pointer.
/// This function is mostly intended for places where panic is not an option
pub(crate) fn try_image_handle() -> Option<NonNull<crate::ffi::c_void>> {
    NonNull::new(GLOBALS.get()?.1.load(Ordering::Acquire))
}
