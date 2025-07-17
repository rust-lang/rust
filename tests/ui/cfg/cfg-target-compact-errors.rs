//@ check-fail

#![feature(cfg_target_compact)]

#[cfg(target(o::o))]
//~^ ERROR malformed `cfg` attribute input
fn one() {}

#[cfg(target(os = 8))]
//~^ ERROR malformed `cfg` attribute input
fn two() {}

#[cfg(target(os = "linux", pointer(width = "64")))]
//~^ ERROR malformed `cfg` attribute input
fn three() {}

#[cfg(target(true))]
//~^ ERROR malformed `cfg` attribute input
fn four() {}

#[cfg(target(clippy::os = "linux"))]
//~^ ERROR `cfg` predicate key must be an identifier
fn five() {}

fn main() {}
