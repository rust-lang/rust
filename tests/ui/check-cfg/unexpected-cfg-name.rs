// Check warning for unexpected configuration name
//
//@ check-pass
//@ compile-flags: --check-cfg=cfg() -Z unstable-options

#[cfg(widnows)]
//~^ WARNING unexpected `cfg` condition name
pub fn f() {}

#[cfg(windows)]
pub fn g() {}

pub fn main() {}
