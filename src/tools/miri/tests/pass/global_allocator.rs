#![feature(allocator_api, slice_ptr_get)]

use std::alloc::{Allocator as _, Global, GlobalAlloc, Layout, System};

#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

struct Allocator;

const SIZE: usize = 16 * 7;

unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // use specific size to avoid getting triggered by rt
        if layout.size() == SIZE {
            println!("Allocated!")
        }

        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.size() == SIZE {
            println!("Deallocated!")
        }

        System.dealloc(ptr, layout)
    }
}

fn main() {
    // Below we mix using `Global` and `System`. This is undefined behavior that Miri should
    // detect, but currently it does not. <https://github.com/rust-lang/miri/issues/2686>

    let l = Layout::from_size_align(SIZE, 1).unwrap();
    let ptr = Global.allocate(l).unwrap().as_non_null_ptr(); // allocating with Global...
    unsafe {
        System.deallocate(ptr, l);
    } // ... and deallocating with System.

    let l = Layout::from_size_align(SIZE, 16).unwrap();
    let ptr = System.allocate(l).unwrap().as_non_null_ptr(); // allocating with System...
    unsafe {
        Global.deallocate(ptr, l);
    } // ... and deallocating with Global.
}
