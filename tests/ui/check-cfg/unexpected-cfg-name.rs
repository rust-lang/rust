// Check warning for unexpected configuration name
//
//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags: --check-cfg=cfg()

#[cfg(widnows)]
//~^ WARNING unexpected `cfg` condition name
pub fn f() {}

#[cfg(test)]
//~^ WARNING unexpected `cfg` condition name

#[cfg(windows)]
pub fn g() {}

pub fn main() {}
