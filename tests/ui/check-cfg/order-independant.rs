//@ check-pass
//
//@ no-auto-check-cfg
//@ revisions: values_before values_after
//@ compile-flags: --check-cfg=cfg(values_before,values_after)
//@ [values_before]compile-flags: --check-cfg=cfg(a,values("b")) --check-cfg=cfg(a)
//@ [values_after]compile-flags: --check-cfg=cfg(a) --check-cfg=cfg(a,values("b"))

#[cfg(a)]
fn my_cfg() {}

#[cfg(a = "unk")]
//~^ WARNING unexpected `cfg` condition value
fn my_cfg() {}

fn main() {}
