#![crate_name="s_upper"]
#![crate_type="dylib"]
#![crate_type="rlib"]

// no-prefer-dynamic

pub extern crate m_middle as m;

mod s_upper_core;
pub use s_upper_core::*;
