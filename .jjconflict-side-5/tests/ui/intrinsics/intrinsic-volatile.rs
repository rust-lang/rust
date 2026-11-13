//@ run-pass

#![feature(core_intrinsics)]

use std::intrinsics::*;

pub fn main() {
    unsafe {
        let mut x: Box<u8> = Box::new(0);
        let mut y: Box<u8> = Box::new(0);

        // test volatile load
        assert_eq!(volatile_load(&*x), 0);
        *x = 1;
        assert_eq!(volatile_load(&*x), 1);

        // test volatile store
        volatile_store(&mut *x, 2);
        assert_eq!(*x, 2);

        // test volatile copy memory
        volatile_copy_memory(&mut *y, &*x, 1);
        assert_eq!(*y, 2);

        // test volatile copy non-overlapping memory
        *x = 3;
        volatile_copy_nonoverlapping_memory(&mut *y, &*x, 1);
        assert_eq!(*y, 3);

        // test volatile set memory
        volatile_set_memory(&mut *x, 4, 1);
        assert_eq!(*x, 4);

        // test unaligned volatile load
        let arr: [u8; 3] = [1, 2, 3];
        let ptr = arr[1..].as_ptr() as *const u16;
        assert_eq!(unaligned_volatile_load(ptr), u16::from_ne_bytes([arr[1], arr[2]]));

        // test unaligned volatile store
        let ptr = arr[1..].as_ptr() as *mut u16;
        unaligned_volatile_store(ptr, 0);
        assert_eq!(arr, [1, 0, 0]);
    }
}
