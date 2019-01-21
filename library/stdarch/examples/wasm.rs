//! A simple slab allocator for pages in wasm

#![feature(stdsimd)]
#![cfg(target_arch = "wasm32")]

extern crate core_arch;

use std::ptr;

use core_arch::arch::wasm32::*;

static mut HEAD: *mut *mut u8 = 0 as _;

#[no_mangle]
pub unsafe extern "C" fn page_alloc() -> *mut u8 {
    if !HEAD.is_null() {
        let next = *HEAD;
        let ret = HEAD;
        HEAD = next as *mut _;
        return ret as *mut u8;
    }

    let ret = memory_grow(0, 1);

    // if we failed to allocate a page then return null
    if ret == usize::max_value() {
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
    (page_size() * (memory_size(0) as u32)) as usize
}

fn page_size() -> u32 {
    64 * 1024
}
