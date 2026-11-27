//@ normalize-stderr: "0x(ff)+" -> "<u128::MAX>"

#![feature( const_trait_impl, const_cmp)]

use std::any::TypeId;

const _: () = {
    let a: TypeId = unsafe { std::mem::transmute(u128::MAX) };
    let b: TypeId = unsafe { std::mem::transmute(u128::MAX) };
    assert!(a == b);
    //~^ ERROR: pointer must point to some allocation
};

fn main() {}
