//@revisions: default uniq
//@compile-flags: -Zmiri-tree-borrows
//@[uniq]compile-flags: -Zmiri-unique-is-unique

//! This is NOT intended behavior.
//! We should eventually find a solution so that the version with `Unique` passes too,
//! otherwise `Unique` is more strict than `&mut`!

#![feature(ptr_internals)]

use core::ptr::addr_of_mut;
use core::ptr::Unique;

fn main() {
    let mut data = 0u8;
    let raw = addr_of_mut!(data);
    unsafe {
        raw_children_of_refmut_can_alias(&mut *raw);
        raw_children_of_unique_can_alias(Unique::new_unchecked(raw));

        // Ultimately the intended behavior is that both above tests would
        // succeed.
        std::hint::unreachable_unchecked();
        //~[default]^ ERROR: entering unreachable code
    }
}

unsafe fn raw_children_of_refmut_can_alias(x: &mut u8) {
    let child1 = addr_of_mut!(*x);
    let child2 = addr_of_mut!(*x);
    // We create two raw aliases of `x`: they have the exact same
    // tag and can be used interchangeably.
    child1.write(1);
    child2.write(2);
    child1.write(1);
    child2.write(2);
}

unsafe fn raw_children_of_unique_can_alias(x: Unique<u8>) {
    let child1 = x.as_ptr();
    let child2 = x.as_ptr();
    // Under `-Zmiri-unique-is-unique`, `Unique` accidentally offers more guarantees
    // than `&mut`. Not because it responds differently to accesses but because
    // there is no easy way to obtain a copy with the same tag.
    //
    // The closest (non-hack) attempt is two calls to `as_ptr`.
    // - Without `-Zmiri-unique-is-unique`, independent `as_ptr` calls return pointers
    //   with the same tag that can thus be used interchangeably.
    // - With the current implementation of `-Zmiri-unique-is-unique`, they return cousin
    //   tags with permissions that do not tolerate aliasing.
    // Eventually we should make such aliasing allowed in some situations
    // (e.g. when there is no protector), which will probably involve
    // introducing a new kind of permission.
    child1.write(1);
    child2.write(2);
    //~[uniq]^ ERROR: /write access through .* is forbidden/
    child1.write(1);
    child2.write(2);
}
