//! UEFI-specific extensions to the primitives in `std::env` module

use crate::ffi::c_void;
use crate::ptr::NonNull;
use crate::sync::atomic::{AtomicPtr, Ordering};
use crate::sync::Once;

static GLOBAL_SYSTEM_TABLE: AtomicPtr<c_void> = AtomicPtr::new(crate::ptr::null_mut());
static GLOBAL_IMAGE_HANDLE: AtomicPtr<c_void> = AtomicPtr::new(crate::ptr::null_mut());
pub(crate) static GLOBALS: Once = Once::new();

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
#[unstable(feature = "uefi_std", issue = "100499")]
pub unsafe fn init_globals(handle: NonNull<c_void>, system_table: NonNull<c_void>) {
    GLOBALS.call_once(|| {
        GLOBAL_SYSTEM_TABLE.store(system_table.as_ptr(), Ordering::Release);
        GLOBAL_IMAGE_HANDLE.store(handle.as_ptr(), Ordering::Release);
    })
}

/// Get the SystemTable Pointer.
/// Note: This function panics if the System Table and Image Handle is Not initialized
#[unstable(feature = "uefi_std", issue = "100499")]
pub fn system_table() -> NonNull<c_void> {
    try_system_table().unwrap()
}

/// Get the SystemHandle Pointer.
/// Note: This function panics if the System Table and Image Handle is Not initialized
#[unstable(feature = "uefi_std", issue = "100499")]
pub fn image_handle() -> NonNull<c_void> {
    try_image_handle().unwrap()
}

/// Get the SystemTable Pointer.
/// This function is mostly intended for places where panic is not an option
pub(crate) fn try_system_table() -> Option<NonNull<crate::ffi::c_void>> {
    NonNull::new(GLOBAL_SYSTEM_TABLE.load(Ordering::Acquire))
}

/// Get the SystemHandle Pointer.
/// This function is mostly intended for places where panic is not an option
pub(crate) fn try_image_handle() -> Option<NonNull<crate::ffi::c_void>> {
    NonNull::new(GLOBAL_IMAGE_HANDLE.load(Ordering::Acquire))
}
