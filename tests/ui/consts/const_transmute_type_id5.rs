//! Test that we require an equal TypeId to have the same integer
//! part, even if the provenance matches.

#![feature(const_type_id, const_trait_impl, const_cmp)]

use std::any::TypeId;

const _: () = {
    let a = TypeId::of::<()>();
    let mut b = TypeId::of::<()>();
    unsafe {
        let ptr = &mut b as *mut TypeId as *mut *const ();
        // Copy the ptr at index 0 to index 1
        let val = std::ptr::read(ptr);
        std::ptr::write(ptr.offset(1), val);
    }
    assert!(a == b);
    //~^ ERROR: type_id_eq: one of the TypeId arguments is invalid, chunk 1 of the hash does not match the type it represents
};

fn main() {}
