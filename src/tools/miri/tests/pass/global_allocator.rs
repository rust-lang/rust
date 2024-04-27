#![feature(allocator_api, slice_ptr_get)]

use std::alloc::{Allocator as _, Global, GlobalAlloc, Layout, System};

#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

struct Allocator;

unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // use specific size to avoid getting triggered by rt
        if layout.size() == 123 {
            println!("Allocated!")
        }

        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.size() == 123 {
            println!("Deallocated!")
        }

        System.dealloc(ptr, layout)
    }
}

fn main() {
    // Only okay because we explicitly set a global allocator that uses the system allocator!
    let l = Layout::from_size_align(123, 1).unwrap();
    let ptr = Global.allocate(l).unwrap().as_non_null_ptr(); // allocating with Global...
    unsafe {
        System.deallocate(ptr, l);
    } // ... and deallocating with System.

    let ptr = System.allocate(l).unwrap().as_non_null_ptr(); // allocating with System...
    unsafe {
        Global.deallocate(ptr, l);
    } // ... and deallocating with Global.
}
