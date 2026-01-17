#![crate_name = "reexport_middle"]
#![crate_type = "rlib"]

extern crate reexport_base;

pub use reexport_base::Thing;
pub use reexport_base::create_thing;
