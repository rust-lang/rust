//! Test that we do not ICE and that we do report an error
//! when placing non-TypeId provenance into a TypeId.

#![feature(const_trait_impl, const_cmp)]

use std::any::TypeId;
use std::mem::transmute;

const X: bool = {
    let a = ();
    let id: TypeId = unsafe { transmute([&raw const a; 16 / size_of::<*const ()>()]) };
    id == id
    //~^ ERROR: invalid `TypeId` value: not all bytes carry type id metadata
};

fn main() {}
