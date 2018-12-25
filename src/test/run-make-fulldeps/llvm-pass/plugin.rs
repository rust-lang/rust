#![feature(plugin_registrar, rustc_private)]
#![crate_type = "dylib"]
#![crate_name = "some_plugin"]

extern crate rustc;
extern crate rustc_plugin;

#[link(name = "llvm-function-pass", kind = "static")]
#[link(name = "llvm-module-pass", kind = "static")]
extern {}

use rustc_plugin::registry::Registry;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_llvm_pass("some-llvm-function-pass");
    reg.register_llvm_pass("some-llvm-module-pass");
}
