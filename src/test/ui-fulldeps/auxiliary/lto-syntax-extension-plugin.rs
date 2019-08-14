// force-host

#![feature(plugin_registrar)]
#![feature(rustc_private)]

extern crate rustc;
extern crate rustc_plugin;
extern crate rustc_driver;

use rustc_plugin::Registry;

#[plugin_registrar]
pub fn plugin_registrar(_reg: &mut Registry) {}
