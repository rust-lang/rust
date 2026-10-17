//! Global Allocator for UEFI.
//! Uses [r-efi-alloc](https://crates.io/crates/r-efi-alloc)

use r_efi::protocols::loaded_image;

use crate::alloc::Layout;
use crate::sync::OnceLock;
use crate::sys::pal::helpers;

pub unsafe fn alloc(layout: Layout) -> *mut u8 {
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
        let protocol =
            helpers::image_handle_protocol::<loaded_image::Protocol>(loaded_image::PROTOCOL_GUID)
                .unwrap();
        // Gives allocations the memory type that the data sections were loaded as.
        unsafe { (*protocol.as_ptr()).image_data_type }
    });

    // The caller must ensure non-0 layout
    unsafe { r_efi_alloc::raw::alloc(system_table, layout, *mem_type) }
}

pub unsafe fn dealloc(ptr: *mut u8, layout: Layout) {
    // Do nothing if boot services are not available
    if crate::os::uefi::env::boot_services().is_none() {
        return;
    }

    // If boot services is valid then SystemTable is not null.
    let system_table = crate::os::uefi::env::system_table().as_ptr().cast();
    // The caller must ensure non-0 layout
    unsafe { r_efi_alloc::raw::dealloc(system_table, ptr, layout) }
}

pub unsafe fn realloc(ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    // SAFETY: this is just a `pub` wrapper.
    unsafe { super::realloc_fallback(ptr, layout, new_size) }
}
