// force-host

#![feature(plugin_registrar, rustc_private)]
#![deny(plugin_as_library)] // should have no effect in a plugin crate

extern crate macro_crate_test;
extern crate rustc;
extern crate rustc_plugin;

use rustc_plugin::Registry;

#[plugin_registrar]
pub fn plugin_registrar(_: &mut Registry) { }
