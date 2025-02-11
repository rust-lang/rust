//@ force-host
//@ no-prefer-dynamic

#![crate_type = "proc-macro"]
#![feature(proc_macro_diagnostic)]

extern crate proc_macro;

mod detail {
    use proc_macro::LintId;

    #[proc_macro_lint]
    pub static ambiguous_thing: LintId;
    //~^ statics tagged with `#[proc_macro_lint]` must currently reside in the root of the crate
}
