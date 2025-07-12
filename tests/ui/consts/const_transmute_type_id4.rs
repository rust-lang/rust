#![feature(const_type_id, const_trait_impl, const_cmp)]

use std::any::TypeId;

const _: () = {
    let a = TypeId::of::<()>();
    let mut b = TypeId::of::<()>();
    unsafe {
        let ptr = &mut b as *mut TypeId as *mut *const ();
        std::ptr::write(ptr.offset(0), main as fn() as *const ());
    }
    assert!(a == b);
    //~^ ERROR: type_id_eq: `TypeId` provenance is not a type id
};

fn main() {}
