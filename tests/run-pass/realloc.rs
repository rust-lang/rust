//ignore-windows: Uses POSIX APIs

#![feature(rustc_private)]

use core::{slice, ptr};

extern crate libc;

fn main() {
    unsafe {
        // Use calloc for initialized memory
        let p1 = libc::calloc(20, 1);

        // old size < new size
        let p2 = libc::realloc(p1, 40);
        let slice = slice::from_raw_parts(p2 as *const u8, 20);
        assert_eq!(&slice, &[0_u8; 20]);

        // old size == new size
        let p3 = libc::realloc(p2, 40);
        let slice = slice::from_raw_parts(p3 as *const u8, 20);
        assert_eq!(&slice, &[0_u8; 20]);

        // old size > new size
        let p4 = libc::realloc(p3, 10);
        let slice = slice::from_raw_parts(p4 as *const u8, 10);
        assert_eq!(&slice, &[0_u8; 10]);

        libc::free(p4);
    }

    unsafe {
        let p1 = libc::malloc(20);

        let p2 = libc::realloc(p1, 0);
        assert!(p2.is_null());
    }

    unsafe {
        let p1 = libc::realloc(ptr::null_mut(), 20);
        assert!(!p1.is_null());

        libc::free(p1);
    }
}
