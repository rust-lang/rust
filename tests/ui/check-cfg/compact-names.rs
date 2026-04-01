// This test check that we correctly emit an warning for compact cfg
//
//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags: --check-cfg=cfg()

#![feature(cfg_target_compact)]

#[cfg(target(os = "linux", arch = "arm"))]
pub fn expected() {}

#[cfg(target(os = "linux", architecture = "arm"))]
//~^ WARNING unexpected `cfg` condition name
pub fn unexpected() {}

#[cfg(target(os = "windows", architecture = "arm"))]
//~^ WARNING unexpected `cfg` condition name
pub fn unexpected2() {}

fn main() {}
