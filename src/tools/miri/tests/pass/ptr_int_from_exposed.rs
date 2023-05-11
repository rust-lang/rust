//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
//@compile-flags: -Zmiri-permissive-provenance
#![feature(strict_provenance)]

use std::ptr;

/// Ensure we can expose the address of a pointer that is out-of-bounds
fn ptr_roundtrip_out_of_bounds() {
    let x: i32 = 3;
    let x_ptr = &x as *const i32;

    let x_usize = x_ptr.wrapping_offset(128).expose_addr();

    let ptr = ptr::from_exposed_addr::<i32>(x_usize).wrapping_offset(-128);
    assert_eq!(unsafe { *ptr }, 3);
}

/// Ensure that we can move between allocations after casting back to a ptr
fn ptr_roundtrip_confusion() {
    let x: i32 = 0;
    let y: i32 = 1;

    let x_ptr = &x as *const i32;
    let y_ptr = &y as *const i32;

    let x_usize = x_ptr.expose_addr();
    let y_usize = y_ptr.expose_addr();

    let ptr = ptr::from_exposed_addr::<i32>(y_usize);
    let ptr = ptr.with_addr(x_usize);
    assert_eq!(unsafe { *ptr }, 0);
}

/// Ensure we can cast back a different integer than the one we got when exposing.
fn ptr_roundtrip_imperfect() {
    let x: u8 = 3;
    let x_ptr = &x as *const u8;

    let x_usize = x_ptr.expose_addr() + 128;

    let ptr = ptr::from_exposed_addr::<u8>(x_usize).wrapping_offset(-128);
    assert_eq!(unsafe { *ptr }, 3);
}

/// Ensure that we can roundtrip through a pointer with an address of 0
fn ptr_roundtrip_null() {
    let x = &42;
    let x_ptr = x as *const i32;
    let x_null_ptr = x_ptr.with_addr(0); // addr 0, but still the provenance of x
    let null = x_null_ptr.expose_addr();
    assert_eq!(null, 0);

    let x_null_ptr_copy = ptr::from_exposed_addr::<i32>(null); // just a roundtrip, so has provenance of x (angelically)
    let x_ptr_copy = x_null_ptr_copy.with_addr(x_ptr.addr()); // addr of x and provenance of x
    assert_eq!(unsafe { *x_ptr_copy }, 42);
}

fn main() {
    ptr_roundtrip_out_of_bounds();
    ptr_roundtrip_confusion();
    ptr_roundtrip_imperfect();
    ptr_roundtrip_null();
}
