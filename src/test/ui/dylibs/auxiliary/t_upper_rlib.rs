#![crate_name="t_upper"]
#![crate_type="rlib"]

// no-prefer-dynamic : flag controls both `-C prefer-dynamic` *and* overrides the
// output crate type for this file.

pub extern crate m_middle as m;

mod t_upper_core;
pub use t_upper_core::*;
