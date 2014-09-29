// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// alloc::heap::reallocate test.
//
// Ideally this would be revised to use no_std, but for now it serves
// well enough to reproduce (and illustrate) the bug from #16687.

extern crate alloc;

use alloc::heap;
use std::ptr;

fn main() {
    unsafe {
        assert!(test_triangle());
    }
}

unsafe fn test_triangle() -> bool {
    static COUNT : uint = 16;
    let mut ascend = Vec::from_elem(COUNT, ptr::null_mut());
    let ascend = ascend.as_mut_slice();
    static ALIGN : uint = 1;

    // Checks that `ascend` forms triangle of acending size formed
    // from pairs of rows (where each pair of rows is equally sized),
    // and the elements of the triangle match their row-pair index.
    unsafe fn sanity_check(ascend: &[*mut u8]) {
        for i in range(0u, COUNT / 2) {
            let (p0, p1, size) = (ascend[2*i], ascend[2*i+1], idx_to_size(i));
            for j in range(0u, size) {
                assert_eq!(*p0.offset(j as int), i as u8);
                assert_eq!(*p1.offset(j as int), i as u8);
            }
        }
    }

    static PRINT : bool = false;

    unsafe fn allocate(size: uint, align: uint) -> *mut u8 {
        if PRINT { println!("allocate(size={:u} align={:u})", size, align); }

        let ret = heap::allocate(size, align);

        if PRINT { println!("allocate(size={:u} align={:u}) ret: 0x{:010x}",
                            size, align, ret as uint);
        }

        ret
    }
    unsafe fn deallocate(ptr: *mut u8, size: uint, align: uint) {
        if PRINT { println!("deallocate(ptr=0x{:010x} size={:u} align={:u})",
                            ptr as uint, size, align);
        }

        heap::deallocate(ptr, size, align);
    }
    unsafe fn reallocate(ptr: *mut u8, size: uint, align: uint,
                             old_size: uint) -> *mut u8 {
        if PRINT {
            println!("reallocate(ptr=0x{:010x} size={:u} align={:u} old_size={:u})",
                     ptr as uint, size, align, old_size);
        }

        let ret = heap::reallocate(ptr, size, align, old_size);

        if PRINT {
            println!("reallocate(ptr=0x{:010x} size={:u} align={:u} old_size={:u}) \
                      ret: 0x{:010x}",
                     ptr as uint, size, align, old_size, ret as uint);
        }
        ret
    }

    fn idx_to_size(i: uint) -> uint { (i+1) * 10 }

    // Allocate pairs of rows that form a triangle shape.  (Hope is
    // that at least two rows will be allocated near each other, so
    // that we trigger the bug (a buffer overrun) in an observable
    // way.)
    for i in range(0u, COUNT / 2) {
        let size = idx_to_size(i);
        ascend[2*i]   = allocate(size, ALIGN);
        ascend[2*i+1] = allocate(size, ALIGN);
    }

    // Initialize each pair of rows to distinct value.
    for i in range(0u, COUNT / 2) {
        let (p0, p1, size) = (ascend[2*i], ascend[2*i+1], idx_to_size(i));
        for j in range(0, size) {
            *p0.offset(j as int) = i as u8;
            *p1.offset(j as int) = i as u8;
        }
    }

    sanity_check(ascend.as_slice());
    test_1(ascend); // triangle -> square
    test_2(ascend); // square -> triangle
    test_3(ascend); // triangle -> square
    test_4(ascend); // square -> triangle

    for i in range(0u, COUNT / 2) {
        let size = idx_to_size(i);
        deallocate(ascend[2*i], size, ALIGN);
        deallocate(ascend[2*i+1], size, ALIGN);
    }

    return true;

    // Test 1: turn the triangle into a square (in terms of
    // allocation; initialized portion remains a triangle) by
    // realloc'ing each row from top to bottom, and checking all the
    // rows as we go.
    unsafe fn test_1(ascend: &mut [*mut u8]) {
        let new_size = idx_to_size(COUNT-1);
        for i in range(0u, COUNT / 2) {
            let (p0, p1, old_size) = (ascend[2*i], ascend[2*i+1], idx_to_size(i));
            assert!(old_size < new_size);

            ascend[2*i] = reallocate(p0, new_size, ALIGN, old_size);
            sanity_check(ascend.as_slice());

            ascend[2*i+1] = reallocate(p1, new_size, ALIGN, old_size);
            sanity_check(ascend.as_slice());
        }
    }

    // Test 2: turn the square back into a triangle, top to bottom.
    unsafe fn test_2(ascend: &mut [*mut u8]) {
        let old_size = idx_to_size(COUNT-1);
        for i in range(0u, COUNT / 2) {
            let (p0, p1, new_size) = (ascend[2*i], ascend[2*i+1], idx_to_size(i));
            assert!(new_size < old_size);

            ascend[2*i] = reallocate(p0, new_size, ALIGN, old_size);
            sanity_check(ascend.as_slice());

            ascend[2*i+1] = reallocate(p1, new_size, ALIGN, old_size);
            sanity_check(ascend.as_slice());
        }
    }

    // Test 3: turn triangle into a square, bottom to top.
    unsafe fn test_3(ascend: &mut [*mut u8]) {
        let new_size = idx_to_size(COUNT-1);
        for i in range(0u, COUNT / 2).rev() {
            let (p0, p1, old_size) = (ascend[2*i], ascend[2*i+1], idx_to_size(i));
            assert!(old_size < new_size);

            ascend[2*i+1] = reallocate(p1, new_size, ALIGN, old_size);
            sanity_check(ascend.as_slice());

            ascend[2*i] = reallocate(p0, new_size, ALIGN, old_size);
            sanity_check(ascend.as_slice());
        }
    }

    // Test 4: turn the square back into a triangle, bottom to top.
    unsafe fn test_4(ascend: &mut [*mut u8]) {
        let old_size = idx_to_size(COUNT-1);
        for i in range(0u, COUNT / 2).rev() {
            let (p0, p1, new_size) = (ascend[2*i], ascend[2*i+1], idx_to_size(i));
            assert!(new_size < old_size);

            ascend[2*i+1] = reallocate(p1, new_size, ALIGN, old_size);
            sanity_check(ascend.as_slice());

            ascend[2*i] = reallocate(p0, new_size, ALIGN, old_size);
            sanity_check(ascend.as_slice());
        }
    }
}
