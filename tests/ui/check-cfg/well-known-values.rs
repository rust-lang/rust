// This test check that we lint on non well known values and that we don't lint on well known
// values
//
// check-pass
// compile-flags: --check-cfg=cfg() -Z unstable-options

#[cfg(target_os = "linuz")]
//~^ WARNING unexpected `cfg` condition value
fn target_os_linux_misspell() {}

#[cfg(target_os = "linux")]
fn target_os_linux() {}

#[cfg(target_has_atomic = "0")]
//~^ WARNING unexpected `cfg` condition value
fn target_has_atomic_invalid() {}

#[cfg(target_has_atomic = "8")]
fn target_has_atomic() {}

#[cfg(unix = "aa")]
//~^ WARNING unexpected `cfg` condition value
fn unix_with_value() {}

#[cfg(unix)]
fn unix() {}

#[cfg(miri = "miri")]
//~^ WARNING unexpected `cfg` condition value
fn miri_with_value() {}

#[cfg(miri)]
fn miri() {}

#[cfg(doc = "linux")]
//~^ WARNING unexpected `cfg` condition value
fn doc_with_value() {}

#[cfg(doc)]
fn doc() {}

fn main() {}
