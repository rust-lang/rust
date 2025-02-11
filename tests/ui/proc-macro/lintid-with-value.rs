//@ force-host
//@ no-prefer-dynamic

#![crate_type = "proc-macro"]
#![feature(proc_macro_diagnostic)]

extern crate proc_macro;

use proc_macro::LintId;

#[proc_macro_lint]
pub static ambiguous_thing: LintId = "...";
//~^ a unique LintId value is automatically filled in by `#[proc_macro_lint]`
//~^^ mismatched types
