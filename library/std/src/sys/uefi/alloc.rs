//! Global Allocator for UEFI.
//! Uses [r-efi-alloc](https://crates.io/crates/r-efi-alloc)

use crate::alloc::{handle_alloc_error, GlobalAlloc, Layout, System};

pub(crate) const POOL_ALIGNMENT: usize = 8;

const MEMORY_TYPE: u32 = r_efi::efi::LOADER_DATA;

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let system_table = match crate::os::uefi::env::try_system_table() {
            None => return crate::ptr::null_mut(),
            Some(x) => x.as_ptr() as *mut _,
        };

        if layout.size() > 0 {
            unsafe { r_efi_alloc::raw::alloc(system_table, layout, MEMORY_TYPE) }
        } else {
            layout.dangling().as_ptr()
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let system_table = match crate::os::uefi::env::try_system_table() {
            None => handle_alloc_error(layout),
            Some(x) => x.as_ptr() as *mut _,
        };
        if layout.size() > 0 {
            unsafe { r_efi_alloc::raw::dealloc(system_table, ptr, layout) }
        }
    }
}
