// This test checks that when no features are passed by Cargo we
// suggest adding some in the Cargo.toml instead of vomitting a
// list of all the expected names
//
// check-pass
// rustc-env:CARGO=/usr/bin/cargo
// compile-flags: --check-cfg=cfg() -Z unstable-options
// error-pattern:Cargo.toml

#[cfg(feature = "serde")]
//~^ WARNING unexpected `cfg` condition name
fn ser() {}

fn main() {}
