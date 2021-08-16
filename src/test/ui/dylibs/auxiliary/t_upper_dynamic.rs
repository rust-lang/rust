#![crate_name="t_upper"]
#![crate_type="dylib"]

pub extern crate m_middle as m;

mod t_upper_core;
pub use t_upper_core::*;
