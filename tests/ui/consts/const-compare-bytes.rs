//@ run-pass

#![feature(core_intrinsics, const_cmp)]
use std::cmp::Ordering;
use std::intrinsics::compare_bytes;

fn main() {
    const A: Ordering = unsafe { compare_bytes(1 as *const u8, 2 as *const u8, 0) };
    assert_eq!(A, Ordering::Equal);

    const B: Ordering = unsafe { compare_bytes([1, 2].as_ptr(), [1, 3].as_ptr(), 1) };
    assert_eq!(B, Ordering::Equal);

    const C: Ordering = unsafe { compare_bytes([1, 2, 9].as_ptr(), [1, 3, 8].as_ptr(), 2) };
    assert_eq!(C, Ordering::Less);

    const D: Ordering = unsafe { compare_bytes([1, 3, 8].as_ptr(), [1, 2, 9].as_ptr(), 2) };
    assert_eq!(D, Ordering::Greater);
}
