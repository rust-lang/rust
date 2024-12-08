// Check warning for unexpected cfg
//
//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags: --check-cfg=cfg()

#[cfg(unknown_key = "value")]
//~^ WARNING unexpected `cfg` condition name
pub fn f() {}

fn main() {}
