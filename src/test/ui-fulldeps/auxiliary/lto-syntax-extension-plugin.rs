// force-host

#![feature(plugin_registrar)]
#![feature(rustc_private)]

extern crate rustc_middle;
extern crate rustc_driver;

use rustc_driver::plugin::Registry;

#[plugin_registrar]
pub fn plugin_registrar(_reg: &mut Registry) {}
