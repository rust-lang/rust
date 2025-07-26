//! A simple slab allocator for pages in wasm

#![cfg(target_arch = "wasm32")]

use std::ptr;

use core_arch::arch::wasm32::*;

static mut HEAD: *mut *mut u8 = 0 as _;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn page_alloc() -> *mut u8 {
    unsafe {
        if !HEAD.is_null() {
            let next = *HEAD;
            let ret = HEAD;
            HEAD = next as *mut _;
            return ret as *mut u8;
        }
    }

    let ret = memory_grow(0, 1);

    // if we failed to allocate a page then return null
    if ret == usize::MAX {
        return ptr::null_mut();
    }

    ((ret as u32) * page_size()) as *mut u8
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn page_free(page: *mut u8) {
    let page = page as *mut *mut u8;
    unsafe {
        *page = HEAD as *mut u8;
        HEAD = page;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn memory_used() -> usize {
    (page_size() * (memory_size(0) as u32)) as usize
}

fn page_size() -> u32 {
    64 * 1024
}
