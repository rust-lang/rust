//! Test that we require an equal TypeId to have an integer part that properly
//! reflects the type id hash.

#![feature(const_trait_impl, const_cmp)]

use std::any::TypeId;

const _: () = {
    let mut b = TypeId::of::<()>();
    unsafe {
        let ptr = &mut b as *mut TypeId as *mut *const ();
        // Copy the ptr at index 0 to index 1
        let val = std::ptr::read(ptr);
        std::ptr::write(ptr.offset(1), val);
    }
    assert!(b == b);
    //~^ ERROR: invalid `TypeId` value
};

fn main() {}
