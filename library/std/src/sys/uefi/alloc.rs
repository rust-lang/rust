//! Global Allocator for UEFI.
//! Uses [r-efi-alloc](https://crates.io/crates/r-efi-alloc)

use crate::alloc::{GlobalAlloc, Layout, System};

const MEMORY_TYPE: u32 = r_efi::efi::LOADER_DATA;

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Return null pointer if boot services are not available
        if crate::os::uefi::env::boot_services().is_none() {
            return crate::ptr::null_mut();
        }

        // If boot services is valid then SystemTable is not null.
        let system_table = crate::os::uefi::env::system_table().as_ptr().cast();
        // The caller must ensure non-0 layout
        unsafe { r_efi_alloc::raw::alloc(system_table, layout, MEMORY_TYPE) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Do nothing if boot services are not available
        if crate::os::uefi::env::boot_services().is_none() {
            return;
        }

        // If boot services is valid then SystemTable is not null.
        let system_table = crate::os::uefi::env::system_table().as_ptr().cast();
        // The caller must ensure non-0 layout
        unsafe { r_efi_alloc::raw::dealloc(system_table, ptr, layout) }
    }
}
