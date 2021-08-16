#![crate_name="i_ground"]
#![crate_type="rlib"]

// no-prefer-dynamic : flag controls both `-C prefer-dynamic` *and* overrides the
// output crate type for this file.

pub extern crate a_basement as a;

mod i_ground_core;
pub use i_ground_core::*;
