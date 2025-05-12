//! Global Allocator for UEFI.
//! Uses [r-efi-alloc](https://crates.io/crates/r-efi-alloc)

use r_efi::protocols::loaded_image;

use crate::alloc::{GlobalAlloc, Layout, System};
use crate::sync::OnceLock;
use crate::sys::pal::helpers;

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        static EFI_MEMORY_TYPE: OnceLock<u32> = OnceLock::new();

        // Return null pointer if boot services are not available
        if crate::os::uefi::env::boot_services().is_none() {
            return crate::ptr::null_mut();
        }

        // If boot services is valid then SystemTable is not null.
        let system_table = crate::os::uefi::env::system_table().as_ptr().cast();

        // Each loaded image has an image handle that supports `EFI_LOADED_IMAGE_PROTOCOL`. Thus, this
        // will never fail.
        let mem_type = EFI_MEMORY_TYPE.get_or_init(|| {
            let protocol = helpers::image_handle_protocol::<loaded_image::Protocol>(
                loaded_image::PROTOCOL_GUID,
            )
            .unwrap();
            // Gives allocations the memory type that the data sections were loaded as.
            unsafe { (*protocol.as_ptr()).image_data_type }
        });

        // The caller must ensure non-0 layout
        unsafe { r_efi_alloc::raw::alloc(system_table, layout, *mem_type) }
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
