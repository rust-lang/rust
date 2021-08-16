#![crate_name="z_roof"]
#![crate_type="dylib"]

pub extern crate s_upper as s;
pub extern crate t_upper as t;

mod z_roof_core;
pub use z_roof_core::*;
