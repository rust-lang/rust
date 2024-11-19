// This test checks that we emit the `unexpected_cfgs` lint even in code
// coming from an external macro.

//@ check-pass
//@ no-auto-check-cfg
//@ aux-crate: cfg_macro=cfg_macro.rs
//@ compile-flags: --check-cfg=cfg()

fn main() {
    cfg_macro::my_lib_macro!();
    //~^ WARNING unexpected `cfg` condition name
}
