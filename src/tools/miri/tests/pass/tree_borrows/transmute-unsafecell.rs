//@compile-flags: -Zmiri-tree-borrows

//! Testing `mem::transmute` between types with and without interior mutability.
//! All transmutations should work, as long as we don't do any actual accesses
//! that violate immutability.

use core::cell::UnsafeCell;
use core::mem;

fn main() {
    unsafe {
        ref_to_cell();
        cell_to_ref();
    }
}

// Pretend that the reference has interior mutability.
// Don't actually mutate it though, it will fail because it has a Frozen parent.
unsafe fn ref_to_cell() {
    let x = &42i32;
    let cell_x: &UnsafeCell<i32> = mem::transmute(x);
    let val = *cell_x.get();
    assert_eq!(val, 42);
}

// Forget about the interior mutability of a cell.
unsafe fn cell_to_ref() {
    let x = &UnsafeCell::new(42);
    let ref_x: &i32 = mem::transmute(x);
    let val = *ref_x;
    assert_eq!(val, 42);
}
