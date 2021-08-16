#![crate_name="s_upper"]
#![crate_type="rlib"]

// no-prefer-dynamic : flag controls both `-C prefer-dynamic` *and* overrides the
// output crate type for this file.

pub extern crate m_middle as m;

mod s_upper_core;
pub use s_upper_core::*;
