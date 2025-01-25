// This test checks that we lint on non well known names and that we don't lint on well known names
//
//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags: --check-cfg=cfg() -Zcheck-cfg-all-expected
//@ normalize-stderr: "`, `" -> "`\n`"

#[cfg(list_all_well_known_cfgs)]
//~^ WARNING unexpected `cfg` condition name
fn in_diagnostics() {}

#[cfg(target_oz = "linux")]
//~^ WARNING unexpected `cfg` condition name
fn target_os_misspell() {}

#[cfg(target_os = "linux")]
fn target_os() {}

#[cfg(features = "foo")]
//~^ WARNING unexpected `cfg` condition name
fn feature_misspell() {}

#[cfg(feature = "foo")]
//~^ WARNING unexpected `cfg` condition name
fn feature() {}

#[cfg(uniw)]
//~^ WARNING unexpected `cfg` condition name
fn unix_misspell() {}

#[cfg(unix)]
fn unix() {}

#[cfg(miri)]
fn miri() {}

#[cfg(doc)]
fn doc() {}

fn main() {}
