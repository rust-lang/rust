// force-host

#![feature(plugin_registrar)]
#![feature(rustc_private)]

extern crate rustc;
extern crate rustc_plugin;

use rustc_plugin::Registry;

#[plugin_registrar]
pub fn plugin_registrar(_reg: &mut Registry) {}
