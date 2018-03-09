//! A simple slab allocator for pages in wasm

#![feature(stdsimd)]
#![cfg(target_arch = "wasm32")]

extern crate stdsimd;

use std::ptr;

use stdsimd::arch::wasm32::*;

static mut HEAD: *mut *mut u8 = 0 as _;

#[no_mangle]
pub unsafe extern "C" fn page_alloc() -> *mut u8 {
    if !HEAD.is_null() {
        let next = *HEAD;
        let ret = HEAD;
        HEAD = next as *mut _;
        return ret as *mut u8;
    }

    let ret = grow_memory(1);

    // if we failed to allocate a page then return null
    if ret == -1 {
        return ptr::null_mut();
    }

    ((ret as u32) * page_size()) as *mut u8
}

#[no_mangle]
pub unsafe extern "C" fn page_free(page: *mut u8) {
    let page = page as *mut *mut u8;
    *page = HEAD as *mut u8;
    HEAD = page;
}

#[no_mangle]
pub unsafe extern "C" fn memory_used() -> usize {
    (page_size() * (current_memory() as u32)) as usize
}

fn page_size() -> u32 {
    64 * 1024
}
