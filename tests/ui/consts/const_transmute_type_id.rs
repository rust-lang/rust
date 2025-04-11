#![feature(const_type_id, const_trait_impl)]

use std::any::TypeId;

const _: () = {
    let id = TypeId::of::<u8>();
    let id: u8 = unsafe { (&id as *const TypeId).cast::<u8>().read() };
    //~^ ERROR: unable to turn pointer into integer
};

fn main() {}
