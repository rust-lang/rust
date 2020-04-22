// run-rustfix

#![warn(clippy::mismatched_target_os)]
#![allow(unused)]

#[cfg(linux)]
fn linux() {}

#[cfg(freebsd)]
fn freebsd() {}

#[cfg(dragonfly)]
fn dragonfly() {}

#[cfg(openbsd)]
fn openbsd() {}

#[cfg(netbsd)]
fn netbsd() {}

#[cfg(macos)]
fn macos() {}

#[cfg(ios)]
fn ios() {}

#[cfg(android)]
fn android() {}

#[cfg(all(not(any(windows, linux)), freebsd))]
fn list() {}

// windows is a valid target family, should be ignored
#[cfg(windows)]
fn windows() {}

// correct use, should be ignored
#[cfg(target_os = "freebsd")]
fn freebsd() {}

fn main() {}
