// Request MIR so normalization runs even though the crate has feature errors.
//@ compile-flags: -Znext-solver --emit=mir

#![feature(generic_const_args)]
//~^ ERROR `generic_const_args` requires `min_generic_const_args` to be enabled
#![allow(incomplete_features)]

use std::field::{field_of, Field};
//~^ ERROR use of unstable library feature `field_projections`
//~| ERROR use of unstable library feature `field_projections`

struct Struct {
    b: i64,
}

fn project_ref() {
    <field_of!(Struct, b)>::OFFSET;
    //~^ ERROR use of unstable library feature `field_projections`
    //~| ERROR use of unstable library feature `field_projections`
}
//~^ ERROR `main` function not found
