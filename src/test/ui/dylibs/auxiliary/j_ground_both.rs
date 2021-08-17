#![crate_name="j_ground"]
#![crate_type="dylib"]
#![crate_type="rlib"]

// no-prefer-dynamic

pub extern crate a_basement as a;

mod j_ground_core;
pub use j_ground_core::*;
