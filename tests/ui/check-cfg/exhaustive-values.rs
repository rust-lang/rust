// Check warning for unexpected cfg value
//
//@ check-pass
//@ no-auto-check-cfg
//@ revisions: empty_cfg without_names
//@ [empty_cfg]compile-flags: --check-cfg=cfg()
//@ [without_names]compile-flags: --check-cfg=cfg(any())

#[cfg(test = "value")]
//~^ WARNING unexpected `cfg` condition value
pub fn f() {}

fn main() {}
