// Tests that it is possible to create a global allocator in a submodule, rather than in the crate
// root.

extern crate alloc;

use std::{
    alloc::{GlobalAlloc, Layout},
    ptr,
};

struct MyAlloc;

unsafe impl GlobalAlloc for MyAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ptr::null_mut()
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {}
}

mod submod {
    use super::MyAlloc;

    #[global_allocator]
    static MY_HEAP: MyAlloc = MyAlloc; //~ ERROR global_allocator
}

fn main() {}
