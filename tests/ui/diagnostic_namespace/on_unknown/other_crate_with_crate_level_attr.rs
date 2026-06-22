//@ aux-crate:other=other_crate_level.rs

#![crate_type = "lib"]
extern crate other;

pub use ::other::nothing;
//~^ ERROR you silly, the crate `other_crate_level` is empty
//~| NOTE unresolved import `other::nothing`
//~| NOTE no `nothing` in the root
