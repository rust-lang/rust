//@ force-host
//@ no-prefer-dynamic

#![crate_type = "proc-macro"]
#![feature(proc_macro_diagnostic)]

extern crate proc_macro;

use proc_macro::LintId;

#[proc_macro_lint]
static ambiguous_thing: LintId;
//~^ statics tagged with `#[proc_macro_lint]` must be `pub`
