#![feature(allocator_api)]

use std::ptr::NonNull;
use std::alloc::{Global, AllocRef, Layout, System};
use std::slice;

fn check_alloc<T: AllocRef>(mut allocator: T) { unsafe {
    for &align in &[4, 8, 16, 32] {
        let layout = Layout::from_size_align(20, align).unwrap();

        for _ in 0..32 {
            let a = allocator.alloc(layout).unwrap().0;
            assert_eq!(a.as_ptr() as usize % align, 0, "pointer is incorrectly aligned");
            allocator.dealloc(a, layout);
        }

        let p1 = allocator.alloc_zeroed(layout).unwrap().0;
        assert_eq!(p1.as_ptr() as usize % align, 0, "pointer is incorrectly aligned");

        let p2 = allocator.realloc(p1, layout, 40).unwrap().0;
        let layout = Layout::from_size_align(40, align).unwrap();
        assert_eq!(p2.as_ptr() as usize % align, 0, "pointer is incorrectly aligned");
        let slice = slice::from_raw_parts(p2.as_ptr(), 20);
        assert_eq!(&slice, &[0_u8; 20]);

        // old size == new size
        let p3 = allocator.realloc(p2, layout, 40).unwrap().0;
        assert_eq!(p3.as_ptr() as usize % align, 0, "pointer is incorrectly aligned");
        let slice = slice::from_raw_parts(p3.as_ptr(), 20);
        assert_eq!(&slice, &[0_u8; 20]);

        // old size > new size
        let p4 = allocator.realloc(p3, layout, 10).unwrap().0;
        let layout = Layout::from_size_align(10, align).unwrap();
        assert_eq!(p4.as_ptr() as usize % align, 0, "pointer is incorrectly aligned");
        let slice = slice::from_raw_parts(p4.as_ptr(), 10);
        assert_eq!(&slice, &[0_u8; 10]);

        allocator.dealloc(p4, layout);
    }
} }

fn check_align_requests<T: AllocRef>(mut allocator: T) {
    for &size in &[2, 8, 64] { // size less than and bigger than alignment
        for &align in &[4, 8, 16, 32] { // Be sure to cover less than and bigger than `MIN_ALIGN` for all architectures
            let iterations = 32;
            unsafe {
                let pointers: Vec<_> = (0..iterations).map(|_| {
                    allocator.alloc(Layout::from_size_align(size, align).unwrap()).unwrap().0
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
    }
}

fn global_to_box() {
    type T = [i32; 4];
    let l = Layout::new::<T>();
    // allocate manually with global allocator, then turn into Box and free there
    unsafe {
        let ptr = Global.alloc(l).unwrap().0.as_ptr() as *mut T;
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
    check_alloc(System);
    check_alloc(Global);
    check_align_requests(System);
    check_align_requests(Global);
    global_to_box();
    box_to_global();
}
