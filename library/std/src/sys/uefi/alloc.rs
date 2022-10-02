//! Global Allocator for UEFI.
//! Uses `EFI_BOOT_SERVICES.AllocatePool()` and `EFI_BOOT_SERVICES.FreePool()`.
//! Takes a lot of inspiration from Windows allocator for Alignment > 8.

use crate::alloc::{GlobalAlloc, Layout, System};
use crate::ptr;
use crate::sys::uefi::common;

pub(crate) const POOL_ALIGNMENT: usize = 8;

const MEMORY_TYPE: u32 = r_efi::efi::LOADER_DATA;

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let align = layout.align();
        let size = layout.size();

        // Return NULL pointer if boot_services pointer cannot be obtained. The only time this
        // should happen is if SystemTable has not been initialized
        let boot_services = match common::try_boot_services() {
            Some(x) => x,
            None => return ptr::null_mut(),
        };

        let allocate_pool_ptr = unsafe { (*boot_services.as_ptr()).allocate_pool };

        let mut ptr: *mut crate::ffi::c_void = ptr::null_mut();
        let aligned_size = align_size(size, align);

        let r = (allocate_pool_ptr)(MEMORY_TYPE, aligned_size, &mut ptr);

        if r.is_error() || ptr.is_null() {
            return ptr::null_mut();
        }

        unsafe { align_ptr(ptr.cast(), align) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let boot_services = match common::try_boot_services() {
            Some(x) => x,
            None => return,
        };

        let free_pool_ptr = unsafe { (*boot_services.as_ptr()).free_pool };

        let ptr = unsafe { unalign_ptr(ptr, layout.align()) };
        let r = (free_pool_ptr)(ptr.cast());

        assert!(!r.is_error());
    }
}

#[inline]
fn align_size(size: usize, align: usize) -> usize {
    if align > POOL_ALIGNMENT {
        // Allocate extra padding in order to be able to satisfy the alignment.
        size + align
    } else {
        size
    }
}

#[repr(C)]
struct Header(*mut u8);

#[inline]
unsafe fn align_ptr(ptr: *mut u8, align: usize) -> *mut u8 {
    if align > POOL_ALIGNMENT {
        let offset = ptr.align_offset(align);

        // SAFETY: `MIN_ALIGN` <= `offset` <= `layout.align()` and the size of the allocated
        // block is `layout.align() + layout.size()`. `aligned` will thus be a correctly aligned
        // pointer inside the allocated block with at least `layout.size()` bytes after it and at
        // least `MIN_ALIGN` bytes of padding before it.
        let aligned = unsafe { ptr.add(offset) };

        // SAFETY: Because the size and alignment of a header is <= `MIN_ALIGN` and `aligned`
        // is aligned to at least `MIN_ALIGN` and has at least `MIN_ALIGN` bytes of padding before
        // it, it is safe to write a header directly before it.
        unsafe { crate::ptr::write((aligned as *mut Header).offset(-1), Header(ptr)) };

        aligned
    } else {
        ptr
    }
}

#[inline]
unsafe fn unalign_ptr(ptr: *mut u8, align: usize) -> *mut u8 {
    if align > POOL_ALIGNMENT {
        // SAFETY: Because of the contract of `System`, `ptr` is guaranteed to be non-null
        // and have a header readable directly before it.
        unsafe { crate::ptr::read((ptr as *mut Header).offset(-1)).0 }
    } else {
        ptr
    }
}
