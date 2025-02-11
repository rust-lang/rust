#![feature(cfg_boolean_literals)]

#[cfg(os_version_min("macos", "11.0"))]
//~^ ERROR `cfg(os_version_min)` is experimental and subject to change
fn foo1() {}

#[cfg(os_version_min("macos", 1.20))] //~ ERROR: expected a version literal
//~^ ERROR `cfg(os_version_min)` is experimental and subject to change
fn foo2() {}

// No warning if cfg'd away.
#[cfg_attr(false, cfg(os_version_min("macos", false)))]
fn foo3() {}

fn main() {
    if cfg!(os_version_min("macos", "11.0")) {}
    //~^ ERROR `cfg(os_version_min)` is experimental and subject to change
}
