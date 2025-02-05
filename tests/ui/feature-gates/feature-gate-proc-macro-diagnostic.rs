//@ force-host
//@ no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::LintId;
//~^ use of unstable library feature `proc_macro_diagnostic`

#[proc_macro_warning]
//~^ the `#[proc_macro_warning]` attribute is an experimental feature
//~^^ use of unstable library feature `proc_macro_diagnostic`
pub static ambiguous_thing: LintId;
//~^ use of unstable library feature `proc_macro_diagnostic`
