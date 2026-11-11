//@ check-fail

#![feature(cfg_target_compact)]

#[cfg(target(o::o))]
//~^ ERROR malformed `cfg` attribute input
//~| NOTE expected a valid identifier here
//~| NOTE for more information, visit
//~| ERROR malformed `cfg` attribute input
//~| NOTE expected this to be of the form `... = "..."`
//~| NOTE for more information, visit
fn one() {}

#[cfg(target(os = 8))]
//~^ ERROR malformed `cfg` attribute input
//~| NOTE expected a string literal here
//~| NOTE for more information, visit
fn two() {}

#[cfg(target(os = "linux", pointer(width = "64")))]
//~^ ERROR malformed `cfg` attribute input
//~| NOTE expected this to be of the form `... = "..."`
//~| NOTE for more information, visit
fn three() {}

#[cfg(target(true))]
//~^ ERROR malformed `cfg` attribute input
//~| NOTE expected this to be of the form `... = "..."`
//~| NOTE for more information, visit
fn four() {}

#[cfg(target(clippy::os = "linux"))]
//~^ ERROR malformed `cfg` attribute input
//~| NOTE for more information, visit
//~| NOTE expected a valid identifier here
fn five() {}

fn main() {}
