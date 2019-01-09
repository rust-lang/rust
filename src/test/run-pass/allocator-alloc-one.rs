#![allow(stable_features)]

#![feature(allocator_api, nonnull)]

use std::alloc::{Alloc, Global, Layout, handle_alloc_error};

fn main() {
    unsafe {
        let ptr = Global.alloc_one::<i32>().unwrap_or_else(|_| {
            handle_alloc_error(Layout::new::<i32>())
        });
        *ptr.as_ptr() = 4;
        assert_eq!(*ptr.as_ptr(), 4);
        Global.dealloc_one(ptr);
    }
}
