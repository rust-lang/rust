//@ aux-crate:other=other.rs

#![crate_type = "lib"]
extern crate other;

pub use ::other::empty::something;
//~^ ERROR you silly, this module is empty
//~| NOTE unresolved import `other::empty::something`
//~| NOTE no `something` in `empty`
