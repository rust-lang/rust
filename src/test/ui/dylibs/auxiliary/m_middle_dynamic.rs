#![crate_name="m_middle"]
#![crate_type="dylib"]

pub extern crate i_ground as i;
pub extern crate j_ground as j;

mod m_middle_core;
pub use m_middle_core::*;
