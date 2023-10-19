// This test checks that we lint on non well known names and that we don't lint on well known names
//
// check-pass
// compile-flags: --check-cfg=cfg() -Z unstable-options

#[cfg(target_oz = "linux")]
//~^ WARNING unexpected `cfg` condition name
fn target_os_misspell() {}

#[cfg(target_os = "linux")]
fn target_os() {}

#[cfg(features = "foo")]
//~^ WARNING unexpected `cfg` condition name
fn feature_misspell() {}

#[cfg(feature = "foo")]
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
