//! Test that all bytes of a TypeId must have the
//! TypeId marker provenance.

#![feature( const_trait_impl, const_cmp)]

use std::any::TypeId;

const _: () = {
    let a = TypeId::of::<()>();
    let mut b = TypeId::of::<()>();
    unsafe {
        let ptr = &mut b as *mut TypeId as *mut usize;
        std::ptr::write(ptr.offset(1), 999);
    }
    assert!(a == b);
    //~^ ERROR: pointer must point to some allocation
};

fn main() {}
