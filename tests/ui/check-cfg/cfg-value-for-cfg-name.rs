// #120427
// This test checks that when a single cfg has a value for user's specified name
// suggest to use `#[cfg(target_os = "linux")]` instead of `#[cfg(linux)]`
//
//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags: --check-cfg=cfg()

#[cfg(linux)]
//~^ WARNING unexpected `cfg` condition name: `linux`
fn x() {}

// will not suggest if the cfg has a value
#[cfg(linux = "os-name")]
//~^ WARNING unexpected `cfg` condition name: `linux`
fn y() {}

fn main() {}
