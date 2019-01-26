// force-host

#![feature(plugin_registrar)]
#![feature(rustc_private)]

extern crate rustc_plugin;

#[plugin_registrar]
fn plugin_registrar(reg: &mut rustc_plugin::Registry) {}
