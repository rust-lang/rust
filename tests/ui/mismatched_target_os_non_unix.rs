// run-rustfix

#![warn(clippy::mismatched_target_os)]
#![allow(unused)]

#[cfg(hermit)]
fn hermit() {}

#[cfg(wasi)]
fn wasi() {}

#[cfg(none)]
fn none() {}

// list with conditions
#[cfg(all(not(windows), wasi))]
fn list() {}

// windows is a valid target family, should be ignored
#[cfg(windows)]
fn windows() {}

// correct use, should be ignored
#[cfg(target_os = "hermit")]
fn correct() {}

fn main() {}
