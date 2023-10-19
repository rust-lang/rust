// check-pass
// compile-flags: -Z unstable-options
// compile-flags: --check-cfg=cfg(my_cfg,values("foo")) --check-cfg=cfg(my_cfg,values("bar"))

#[cfg(my_cfg)]
//~^ WARNING unexpected `cfg` condition value
fn my_cfg() {}

#[cfg(my_cfg = "unk")]
//~^ WARNING unexpected `cfg` condition value
fn my_cfg() {}

fn main() {}
