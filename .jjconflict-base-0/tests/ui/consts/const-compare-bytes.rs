//@ run-pass

#![feature(core_intrinsics, const_cmp)]
use std::intrinsics::compare_bytes;

fn main() {
    const A: i32 = unsafe {
        compare_bytes(1 as *const u8, 2 as *const u8, 0)
    };
    assert_eq!(A, 0);

    const B: i32 = unsafe {
        compare_bytes([1, 2].as_ptr(), [1, 3].as_ptr(), 1)
    };
    assert_eq!(B, 0);

    const C: i32 = unsafe {
        compare_bytes([1, 2, 9].as_ptr(), [1, 3, 8].as_ptr(), 2)
    };
    assert!(C < 0);

    const D: i32 = unsafe {
        compare_bytes([1, 3, 8].as_ptr(), [1, 2, 9].as_ptr(), 2)
    };
    assert!(D > 0);
}
