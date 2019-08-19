// force-host

#![feature(plugin_registrar)]
#![feature(rustc_private)]

extern crate rustc;
extern crate rustc_plugin;
extern crate rustc_driver;

use rustc_plugin::Registry;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    // This pass is built in to LLVM.
    //
    // Normally, we would name a pass that was registered through
    // C++ static object constructors in the same .so file as the
    // plugin registrar.
    reg.register_llvm_pass("gvn");
}
