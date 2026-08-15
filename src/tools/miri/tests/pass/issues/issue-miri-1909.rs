//@compile-flags: -Zmiri-permissive-provenance
#![deny(unsafe_op_in_unsafe_fn)]
//! This does some tricky ptr-int-casting.

use core::alloc::{GlobalAlloc, Layout};
use std::alloc::System;

/// # Safety
/// `ptr` must be valid for writes of `len` bytes
unsafe fn volatile_write_zeroize_mem(ptr: *mut u8, len: usize) {
    for i in 0..len {
        // ptr as usize + i can't overflow because `ptr` is valid for writes of `len`
        let ptr_new: *mut u8 = ((ptr as usize) + i) as *mut u8;
        // SAFETY: `ptr` is valid for writes of `len` bytes, so `ptr_new` is valid for a
        // byte write
        unsafe {
            core::ptr::write_volatile(ptr_new, 0u8);
        }
    }
}

pub struct ZeroizeAlloc;

unsafe impl GlobalAlloc for ZeroizeAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: uphold by caller
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // securely wipe the deallocated memory
        // SAFETY: `ptr` is valid for writes of `layout.size()` bytes since it was
        // previously successfully allocated (by the safety assumption on this function)
        // and not yet deallocated
        unsafe {
            volatile_write_zeroize_mem(ptr, layout.size());
        }
        // SAFETY: uphold by caller
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: uphold by caller
        unsafe { System.alloc_zeroed(layout) }
    }
}

#[global_allocator]
static GLOBAL: ZeroizeAlloc = ZeroizeAlloc;

fn main() {
    let layout = Layout::new::<[u8; 16]>();
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    unsafe {
        std::alloc::dealloc(ptr, layout);
    }
}
