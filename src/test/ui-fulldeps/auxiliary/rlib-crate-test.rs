// no-prefer-dynamic

#![crate_type = "rlib"]
#![feature(plugin_registrar, rustc_private)]

extern crate rustc_middle;
extern crate rustc_driver;

use rustc_driver::plugin::Registry;

#[plugin_registrar]
pub fn plugin_registrar(_: &mut Registry) {}
