#![crate_name="i_ground"]
#![crate_type="dylib"]
#![crate_type="rlib"]

// no-prefer-dynamic

pub extern crate a_basement as a;

mod i_ground_core;
pub use i_ground_core::*;
