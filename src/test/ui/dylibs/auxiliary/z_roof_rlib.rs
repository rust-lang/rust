#![crate_name="z_roof"]
#![crate_type="rlib"]

// no-prefer-dynamic : flag controls both `-C prefer-dynamic` *and* overrides the
// output crate type for this file.

pub extern crate s_upper as s;
pub extern crate t_upper as t;

mod z_roof_core;
pub use z_roof_core::*;
