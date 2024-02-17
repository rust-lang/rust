//@ compile-flags: -C debuginfo=2 -C prefer-dynamic

#![crate_type="dylib"]

extern crate a_def_obj;

pub use a_def_obj::Object;
