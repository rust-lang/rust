// Regression test for #86262. Previously, the code below would incorrectly
// give E0596 (cannot borrow `this.slot` as mutable) because autoborrowing
// did not handle coercions to mutable raw pointers correctly.

// check-pass
#![crate_type="lib"]
#![allow(unused_mut)]

use std::mem::ManuallyDrop;

pub struct RefWrapper<'a, T: ?Sized> {
    slot: &'a mut T,
}

impl<'a, T: ?Sized> RefWrapper<'a, T> {
    pub fn into_raw(this: Self) -> *mut T {
        let mut this = ManuallyDrop::new(this);
        this.slot as *mut T /* location of former error */
    }
}
