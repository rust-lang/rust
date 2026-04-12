//! ThingOS memory allocator.
//!
//! The global allocator maps anonymous memory regions via the `SYS_VM_MAP`
//! system call and releases them with `SYS_VM_UNMAP`.

use crate::alloc::{GlobalAlloc, Layout, System};
use crate::sys::pal::common::{SYS_VM_MAP, SYS_VM_UNMAP, VM_MAP_ANON_RW, cvt, raw_syscall6};

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SYS_VM_MAP(size, align, flags) -> ptr
        let ret = unsafe {
            raw_syscall6(
                SYS_VM_MAP,
                layout.size() as u64,
                layout.align() as u64,
                VM_MAP_ANON_RW,
                0,
                0,
                0,
            )
        };
        if ret <= 0 { core::ptr::null_mut() } else { ret as usize as *mut u8 }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SYS_VM_MAP zeroes memory by default on ThingOS.
        unsafe { self.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe {
            raw_syscall6(SYS_VM_UNMAP, ptr as u64, layout.size() as u64, 0, 0, 0, 0);
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // ThingOS does not expose a resize syscall; fall back to alloc + copy.
        unsafe {
            let new_layout =
                Layout::from_size_align_unchecked(new_size, layout.align());
            let new_ptr = self.alloc(new_layout);
            if !new_ptr.is_null() {
                let copy_len = core::cmp::min(layout.size(), new_size);
                core::ptr::copy_nonoverlapping(ptr, new_ptr, copy_len);
                self.dealloc(ptr, layout);
            }
            new_ptr
        }
    }
}
