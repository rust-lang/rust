// error-pattern:cargo-clippy
#![feature(plugin_registrar)]
#![feature(rustc_private)]
#![allow(unknown_lints)]

extern crate rustc_plugin;
use rustc_plugin::Registry;

extern crate clippy_lints;

pub use clippy_lints::*;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    register_plugins(reg);
}

// only exists to let the dogfood integration test works.
// Don't run clippy as an executable directly
#[allow(dead_code, print_stdout)]
fn main() {
    panic!("Please use the cargo-clippy executable");
}
