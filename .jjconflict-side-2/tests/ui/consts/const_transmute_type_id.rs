#![feature(const_trait_impl, const_cmp)]

use std::any::TypeId;

const _: () = {
    let id = TypeId::of::<u8>();
    let id: u8 = unsafe { (&raw const id).cast::<u8>().read() };
    //~^ ERROR: unable to turn pointer into integer
};

fn main() {}
