//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags: --check-cfg=cfg(my_cfg,values("foo")) --check-cfg=cfg(my_cfg,values("bar"))
//@ compile-flags: --check-cfg=cfg(my_cfg,values())

#[cfg(my_cfg)]
//~^ WARNING unexpected `cfg` condition value
fn my_cfg() {}

#[cfg(my_cfg = "unk")]
//~^ WARNING unexpected `cfg` condition value
fn my_cfg() {}

#[cfg(any(my_cfg = "foo", my_cfg = "bar"))]
fn foo_and_bar() {}

fn main() {}
