//ignore-windows: inspects allocation base address on Windows

#![feature(allocator_api)]

use std::ptr::NonNull;
use std::alloc::{Global, Alloc, Layout, System};

fn check_overalign_requests<T: Alloc>(mut allocator: T) {
    let size = 8;
    // Greater than `size`.
    let align = 16;
    // Miri is deterministic; no need to try many times.
    let iterations = 1;
    unsafe {
        let pointers: Vec<_> = (0..iterations).map(|_| {
            allocator.alloc(Layout::from_size_align(size, align).unwrap()).unwrap()
        }).collect();
        for &ptr in &pointers {
            assert_eq!((ptr.as_ptr() as usize) % align, 0,
                       "Got a pointer less aligned than requested")
        }

        // Clean up.
        for &ptr in &pointers {
            allocator.dealloc(ptr, Layout::from_size_align(size, align).unwrap())
        }
    }
}

fn global_to_box() {
    type T = [i32; 4];
    let l = Layout::new::<T>();
    // allocate manually with global allocator, then turn into Box and free there
    unsafe {
        let ptr = Global.alloc(l).unwrap().as_ptr() as *mut T;
        let b = Box::from_raw(ptr);
        drop(b);
    }
}

fn box_to_global() {
    type T = [i32; 4];
    let l = Layout::new::<T>();
    // allocate with the Box, then deallocate manually with global allocator
    unsafe {
        let b = Box::new(T::default());
        let ptr = Box::into_raw(b);
        Global.dealloc(NonNull::new(ptr as *mut u8).unwrap(), l);
    }
}

fn main() {
    check_overalign_requests(System);
    check_overalign_requests(Global);
    global_to_box();
    box_to_global();
}
