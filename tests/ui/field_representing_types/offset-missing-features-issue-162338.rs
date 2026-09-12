// Request MIR so normalization runs even though the crate has feature errors.
//@ compile-flags: -Znext-solver --emit=mir

#![feature(gca_const_items)]
//~^ ERROR `gca_const_items` requires `gca_min_const_items` to be enabled
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
