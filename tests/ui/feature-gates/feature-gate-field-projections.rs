#![allow(dead_code)]

use std::field::Field; //~ ERROR: use of unstable library feature `field_projections` [E0658]
use std::ptr;

fn project_ref<F: Field>(
    //~^ ERROR: use of unstable library feature `field_projections` [E0658]
    r: &F::Base, //~ ERROR: use of unstable library feature `field_projections` [E0658]
) -> &F::Type
//~^ ERROR: use of unstable library feature `field_projections` [E0658]
where
    F::Type: Sized, //~ ERROR: use of unstable library feature `field_projections` [E0658]
{
    unsafe { &*ptr::from_ref(r).byte_add(F::OFFSET).cast() } //~ ERROR: use of unstable library feature `field_projections` [E0658]
}

fn main() {}
