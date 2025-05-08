//@compile-flags: -Zmiri-tree-borrows
#![feature(allocator_api)]

use std::{mem, ptr};

// Test various tree-borrows-specific things
// (i.e., these do not work the same under SB).
fn main() {
    aliasing_read_only_mutable_refs();
    string_as_mut_ptr();
    two_mut_protected_same_alloc();
    direct_mut_to_const_raw();
    local_addr_of_mut();
    returned_mut_is_usable();
}

#[allow(unused_assignments)]
fn local_addr_of_mut() {
    let mut local = 0;
    let ptr = ptr::addr_of_mut!(local);
    // In SB, `local` and `*ptr` would have different tags, but in TB they have the same tag.
    local = 1;
    unsafe { *ptr = 2 };
    local = 3;
    unsafe { *ptr = 4 };
}

// Tree Borrows has no issue with several mutable references existing
// at the same time, as long as they are used only immutably.
// I.e. multiple Reserved can coexist.
pub fn aliasing_read_only_mutable_refs() {
    unsafe {
        let base = &mut 42u64;
        let r1 = &mut *(base as *mut u64);
        let r2 = &mut *(base as *mut u64);
        let _l = *r1;
        let _l = *r2;
    }
}

pub fn string_as_mut_ptr() {
    // This errors in Stacked Borrows since as_mut_ptr restricts the provenance,
    // but with Tree Borrows it should work.
    unsafe {
        let mut s = String::from("hello");
        s.reserve(1); // make the `str` that `s` derefs to not cover the entire `s`.

        // Prevent automatically dropping the String's data
        let mut s = mem::ManuallyDrop::new(s);

        let ptr = s.as_mut_ptr();
        let len = s.len();
        let capacity = s.capacity();

        let s = String::from_raw_parts(ptr, len, capacity);

        assert_eq!(String::from("hello"), s);
    }
}

// This function checks that there is no issue with having two mutable references
// from the same allocation both under a protector.
// This is safe code, it must absolutely not be UB.
// This test failing is a symptom of forgetting to check that only initialized
// locations can cause protector UB.
fn two_mut_protected_same_alloc() {
    fn write_second(_x: &mut u8, y: &mut u8) {
        // write through `y` will make some locations of `x` (protected)
        // become Disabled. Those locations are outside of the range on which
        // `x` is initialized, and the protector must not trigger.
        *y = 1;
    }

    let mut data = (0u8, 1u8);
    write_second(&mut data.0, &mut data.1);
}

// This checks that a reborrowed mutable reference returned from a function
// is actually writeable.
// The fact that this is not obvious is due to the addition of
// implicit reads on function exit that might freeze the return value.
fn returned_mut_is_usable() {
    fn reborrow(x: &mut u8) -> &mut u8 {
        let y = &mut *x;
        // Activate the reference so that it is vulnerable to foreign reads.
        *y = *y;
        y
        // An implicit read through `x` is inserted here.
    }
    let mut data = 0;
    let x = &mut data;
    let y = reborrow(x);
    *y = 1;
}

// Make sure that coercing &mut T to *const T produces a writeable pointer.
fn direct_mut_to_const_raw() {
    let x = &mut 0;
    let y: *const i32 = x;
    unsafe {
        *(y as *mut i32) = 1;
    }
    assert_eq!(*x, 1);
}
