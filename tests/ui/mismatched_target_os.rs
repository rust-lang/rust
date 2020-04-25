// run-rustfix

#![warn(clippy::mismatched_target_os)]
#![allow(unused)]

// unix

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

#[cfg(emscripten)]
fn emscripten() {}

#[cfg(fuchsia)]
fn fuchsia() {}

#[cfg(haiku)]
fn haiku() {}

#[cfg(illumos)]
fn illumos() {}

#[cfg(l4re)]
fn l4re() {}

#[cfg(redox)]
fn redox() {}

#[cfg(solaris)]
fn solaris() {}

#[cfg(vxworks)]
fn vxworks() {}

// non-unix

#[cfg(cloudabi)]
fn cloudabi() {}

#[cfg(hermit)]
fn hermit() {}

#[cfg(wasi)]
fn wasi() {}

#[cfg(none)]
fn none() {}

// list with conditions
#[cfg(all(not(any(windows, linux)), freebsd))]
fn list() {}

// windows is a valid target family, should be ignored
#[cfg(windows)]
fn windows() {}

// correct use, should be ignored
#[cfg(target_os = "freebsd")]
fn freebsd() {}

fn main() {}
