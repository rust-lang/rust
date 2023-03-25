//! Global Allocator for UEFI.
//! Uses [r-efi-alloc](https://crates.io/crates/r-efi-alloc)

use crate::alloc::{handle_alloc_error, GlobalAlloc, Layout, System};

const MEMORY_TYPE: u32 = r_efi::efi::LOADER_DATA;

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Return null pointer if boot services are not available
        if crate::os::uefi::env::boot_services().is_none() {
            return crate::ptr::null_mut();
        }

        let system_table = match crate::os::uefi::env::try_system_table() {
            None => return crate::ptr::null_mut(),
            Some(x) => x.as_ptr() as *mut _,
        };

        // The caller must ensure non-0 layout
        unsafe { r_efi_alloc::raw::alloc(system_table, layout, MEMORY_TYPE) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Do nothing if boot services are not available
        if crate::os::uefi::env::boot_services().is_none() {
            return;
        }

        let system_table = match crate::os::uefi::env::try_system_table() {
            None => handle_alloc_error(layout),
            Some(x) => x.as_ptr() as *mut _,
        };
        // The caller must ensure non-0 layout
        unsafe { r_efi_alloc::raw::dealloc(system_table, ptr, layout) }
    }
}
