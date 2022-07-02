//! UEFI-specific extensions to the primitives in `std::env` module

use super::raw::{BootServices, SystemTable};
use crate::ffi::c_void;
use crate::ptr::NonNull;
use crate::sync::atomic::{AtomicPtr, Ordering};

static GLOBAL_SYSTEM_TABLE: AtomicPtr<SystemTable> = AtomicPtr::new(crate::ptr::null_mut());
static GLOBAL_SYSTEM_HANDLE: AtomicPtr<c_void> = AtomicPtr::new(crate::ptr::null_mut());

#[unstable(feature = "uefi_std", issue = "none")]
/// Initializes Global Atomic Pointers to SystemTable and Handle.
/// Should only be called once in the program execution under normal circumstances.
/// The caller should ensure that the pointers are valid.
pub fn init_globals(handle: NonNull<c_void>, system_table: NonNull<SystemTable>) {
    GLOBAL_SYSTEM_TABLE.store(system_table.as_ptr(), Ordering::SeqCst);
    GLOBAL_SYSTEM_HANDLE.store(handle.as_ptr(), Ordering::SeqCst);
}

#[unstable(feature = "uefi_std", issue = "none")]
/// This function returns error if SystemTable pointer is null
/// On success, the returned pointer is guarenteed to be non-null.
pub fn get_system_table() -> Option<NonNull<SystemTable>> {
    NonNull::new(GLOBAL_SYSTEM_TABLE.load(Ordering::SeqCst))
}

#[unstable(feature = "uefi_std", issue = "none")]
/// This function returns error if SystemHandle pointer is null
/// On success, the returned pointer is guarenteed to be non-null.
pub fn get_system_handle() -> Option<NonNull<c_void>> {
    NonNull::new(GLOBAL_SYSTEM_HANDLE.load(Ordering::SeqCst))
}

#[unstable(feature = "uefi_std", issue = "none")]
/// On success, the returned pointer is guarenteed to be non-null.
/// SAFETY: Do not cache the returned pointer unless you are sure it will remain valid.
pub fn get_boot_services() -> Option<NonNull<BootServices>> {
    let system_table = get_system_table()?;
    let boot_services = unsafe { (*system_table.as_ptr()).boot_services };
    NonNull::new(boot_services)
}
