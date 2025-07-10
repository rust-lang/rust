#![feature(const_type_id, const_trait_impl)]

use std::any::TypeId;

const _: () = {
    let a = TypeId::of::<()>();
    let mut b = TypeId::of::<()>();
    unsafe {
        let ptr = &mut b as *mut TypeId as *mut usize;
        std::ptr::write(ptr.offset(1), 999);
    }
    assert!(a == b);
    //~^ ERROR: one of the TypeId arguments is invalid, the hash does not match the type it represents
};

fn main() {}
