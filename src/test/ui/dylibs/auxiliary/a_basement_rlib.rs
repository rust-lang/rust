#![crate_name="a_basement"]
#![crate_type="rlib"]

// no-prefer-dynamic : flag controls both `-C prefer-dynamic` *and* overrides the
// output crate type for this file.

mod a_basement_core;
pub use a_basement_core::*;
