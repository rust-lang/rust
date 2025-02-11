//@ force-host
//@ no-prefer-dynamic

#![crate_type = "proc-macro"]
#![feature(proc_macro_diagnostic)]

extern crate proc_macro;

#[proc_macro_lint]
pub static ambiguous_thing: String;
//~^ wrong type for proc macro lint id
