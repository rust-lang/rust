// Check warning for unexpected cfg in the code.
//
//@ check-pass
//@ no-auto-check-cfg
//@ revisions: empty_cfg feature full
//@ [empty_cfg]compile-flags: --check-cfg=cfg()
//@ [feature]compile-flags: --check-cfg=cfg(feature,values("std"))
//@ [full]compile-flags: --check-cfg=cfg(feature,values("std")) --check-cfg=cfg()

#[cfg(unknown_key = "value")]
//~^ WARNING unexpected `cfg` condition name
pub fn f() {}

#[cfg(target_vendor = "value")]
//~^ WARNING unexpected `cfg` condition value
pub fn f() {}

#[cfg(feature = "unk")]
//[empty_cfg]~^ WARNING unexpected `cfg` condition name
//[feature]~^^ WARNING unexpected `cfg` condition value
//[full]~^^^ WARNING unexpected `cfg` condition value
pub fn feat() {}

#[cfg(feature = "std")]
//[empty_cfg]~^ WARNING unexpected `cfg` condition name
pub fn feat() {}

#[cfg(windows)]
pub fn win() {}

#[cfg(unix)]
pub fn unix() {}

fn main() {}
