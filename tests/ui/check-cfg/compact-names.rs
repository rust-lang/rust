// This test check that we correctly emit an warning for compact cfg
//
//@ check-pass
//@ compile-flags: --check-cfg=cfg() -Z unstable-options

#![feature(cfg_target_compact)]

#[cfg(target(os = "linux", arch = "arm"))]
pub fn expected() {}

#[cfg(target(os = "linux", architecture = "arm"))]
//~^ WARNING unexpected `cfg` condition name
pub fn unexpected() {}

fn main() {}
