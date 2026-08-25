#![feature(allocator_api, slice_ptr_get)]

use std::alloc::{Allocator, Global, Layout, System};
use std::ptr::NonNull;
use std::slice;

fn check_alloc<T: Allocator>(allocator: T) {
    unsafe {
        for &align in &[4, 8, 16, 32] {
            let layout_20 = Layout::from_size_align(20, align).unwrap();
            let layout_40 = Layout::from_size_align(40, 4 * align).unwrap();
            let layout_10 = Layout::from_size_align(10, align / 2).unwrap();

            for _ in 0..32 {
                let a = allocator.allocate(layout_20).unwrap().as_non_null_ptr();
                assert_eq!(
                    a.as_ptr() as usize % layout_20.align(),
                    0,
                    "pointer is incorrectly aligned",
                );
                allocator.deallocate(a, layout_20);
            }

            let p1 = allocator.allocate_zeroed(layout_20).unwrap().as_non_null_ptr();
            assert_eq!(
                p1.as_ptr() as usize % layout_20.align(),
                0,
                "pointer is incorrectly aligned",
            );
            assert_eq!(*p1.as_ptr(), 0);

            // old size < new size
            let p2 = allocator.grow(p1, layout_20, layout_40).unwrap().as_non_null_ptr();
            assert_eq!(
                p2.as_ptr() as usize % layout_40.align(),
                0,
                "pointer is incorrectly aligned",
            );
            let slice = slice::from_raw_parts(p2.as_ptr(), 20);
            assert_eq!(&slice, &[0_u8; 20]);

            // old size == new size
            let p3 = allocator.grow(p2, layout_40, layout_40).unwrap().as_non_null_ptr();
            assert_eq!(
                p3.as_ptr() as usize % layout_40.align(),
                0,
                "pointer is incorrectly aligned",
            );
            let slice = slice::from_raw_parts(p3.as_ptr(), 20);
            assert_eq!(&slice, &[0_u8; 20]);

            // old size > new size
            let p4 = allocator.shrink(p3, layout_40, layout_10).unwrap().as_non_null_ptr();
            assert_eq!(
                p4.as_ptr() as usize % layout_10.align(),
                0,
                "pointer is incorrectly aligned",
            );
            let slice = slice::from_raw_parts(p4.as_ptr(), 10);
            assert_eq!(&slice, &[0_u8; 10]);

            allocator.deallocate(p4, layout_10);
        }
    }
}

fn check_align_requests<T: Allocator>(allocator: T) {
    #[rustfmt::skip] // https://github.com/rust-lang/rustfmt/issues/3255
    for &size in &[2, 8, 64] { // size less than and bigger than alignment
        for &align in &[4, 8, 16, 32] { // Be sure to cover less than and bigger than `MIN_ALIGN` for all architectures
            let iterations = 32;
            unsafe {
                let pointers: Vec<_> = (0..iterations)
                    .map(|_| {
                        allocator
                            .allocate(Layout::from_size_align(size, align).unwrap())
                            .unwrap()
                            .as_non_null_ptr()
                    })
                    .collect();
                for &ptr in &pointers {
                    assert_eq!(
                        (ptr.as_ptr() as usize) % align,
                        0,
                        "Got a pointer less aligned than requested",
                    )
                }

                // Clean up.
                for &ptr in &pointers {
                    allocator.deallocate(ptr, Layout::from_size_align(size, align).unwrap())
                }
            }
        }
    };
}

fn global_to_box() {
    type T = [i32; 4];
    let l = Layout::new::<T>();
    // allocate manually with global allocator, then turn into Box and free there
    unsafe {
        let ptr = Global.allocate(l).unwrap().as_non_null_ptr().as_ptr() as *mut T;
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
        Global.deallocate(NonNull::new(ptr as *mut u8).unwrap(), l);
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
