// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

use alloc::{GlobalAlloc, Layout, System};
use cmp;
use ptr;

// The minimum alignment guaranteed by the architecture. This value is used to
// add fast paths for low alignment values.
#[cfg(all(any(target_arch = "x86",
              target_arch = "arm",
              target_arch = "mips",
              target_arch = "powerpc",
              target_arch = "powerpc64",
              target_arch = "asmjs",
              target_arch = "wasm32")))]
pub const MIN_ALIGN: usize = 8;
#[cfg(all(any(target_arch = "x86_64",
              target_arch = "aarch64",
              target_arch = "mips64",
              target_arch = "s390x",
              target_arch = "sparc64")))]
pub const MIN_ALIGN: usize = 16;

pub unsafe fn realloc_fallback(
    alloc: &System,
    ptr: *mut u8,
    old_layout: Layout,
    new_size: usize,
) -> *mut u8 {
    // Docs for GlobalAlloc::realloc require this to be valid:
    let new_layout = Layout::from_size_align_unchecked(new_size, old_layout.align());

    let new_ptr = GlobalAlloc::alloc(alloc, new_layout);
    if !new_ptr.is_null() {
        let size = cmp::min(old_layout.size(), new_size);
        ptr::copy_nonoverlapping(ptr, new_ptr, size);
        GlobalAlloc::dealloc(alloc, ptr, old_layout);
    }
    new_ptr
}
