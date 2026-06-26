// This test checks that we emit the `unexpected_cfgs` lint even in code
// coming from an external macro.

//@ check-pass
//@ no-auto-check-cfg
//@ revisions: cargo rustc
//@ [rustc]unset-rustc-env:CARGO_CRATE_NAME
//@ [cargo]rustc-env:CARGO_CRATE_NAME=foo
//@ aux-crate: cfg_macro=cfg_macro.rs
//@ compile-flags: --check-cfg=cfg(feature,values())

fn main() {
    cfg_macro::my_lib_macro!();
    //~^ WARNING unexpected `cfg` condition name

    cfg_macro::my_lib_macro_value!();
    //~^ WARNING unexpected `cfg` condition value

    cfg_macro::my_lib_macro_feature!();
    //~^ WARNING unexpected `cfg` condition value
}
