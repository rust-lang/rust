#![crate_name="m_middle"]
#![crate_type="rlib"]

// no-prefer-dynamic : flag controls both `-C prefer-dynamic` *and* overrides the
// output crate type for this file.

pub extern crate i_ground as i;
pub extern crate j_ground as j;

mod m_middle_core;
pub use m_middle_core::*;
