#![crate_name="j_ground"]
#![crate_type="rlib"]

// no-prefer-dynamic : flag controls both `-C prefer-dynamic` *and* overrides the
// output crate type for this file.

pub extern crate a_basement as a;

mod j_ground_core;
pub use j_ground_core::*;
