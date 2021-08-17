#![crate_name="a_basement"]
#![crate_type="dylib"]
#![crate_type="rlib"]

// no-prefer-dynamic

mod a_basement_core;
pub use a_basement_core::*;
