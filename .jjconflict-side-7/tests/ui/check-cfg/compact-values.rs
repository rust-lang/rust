// This test check that we correctly emit an warning for compact cfg
//
//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags: --check-cfg=cfg()

#![feature(cfg_target_compact)]

#[cfg(target(os = "linux", arch = "arm"))]
pub fn expected() {}

#[cfg(target(os = "linux", pointer_width = "X"))]
//~^ WARNING unexpected `cfg` condition value
pub fn unexpected() {}

fn main() {}
