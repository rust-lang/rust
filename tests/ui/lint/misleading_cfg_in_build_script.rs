// This test checks the `cfg()` attributes/macros in cargo build scripts.
//
//@ no-auto-check-cfg

#![deny(misleading_cfg_in_build_script)]
#![allow(dead_code)]

#[cfg(windows)]
//~^ ERROR: misleading_cfg_in_build_script
fn unused_windows_fn() {}
#[cfg(not(windows))]
//~^ ERROR: misleading_cfg_in_build_script
fn unused_not_windows_fn() {}
#[cfg(any(windows, feature = "yellow", unix))]
//~^ ERROR: misleading_cfg_in_build_script
fn pink() {}

// Should not lint.
#[cfg(feature = "green")]
fn pink() {}
#[cfg(bob)]
fn bob() {}

fn main() {
    if cfg!(windows) {}
    //~^ ERROR: misleading_cfg_in_build_script
    if cfg!(not(windows)) {}
    //~^ ERROR: misleading_cfg_in_build_script
    if cfg!(target_os = "freebsd") {}
    //~^ ERROR: misleading_cfg_in_build_script
    if cfg!(any(target_os = "freebsd", windows)) {}
    //~^ ERROR: misleading_cfg_in_build_script
    if cfg!(not(any(target_os = "freebsd", windows))) {}
    //~^ ERROR: misleading_cfg_in_build_script
    if cfg!(all(target_os = "freebsd", windows)) {}
    //~^ ERROR: misleading_cfg_in_build_script
    if cfg!(not(all(target_os = "freebsd", windows))) {}
    //~^ ERROR: misleading_cfg_in_build_script
    if cfg!(all(target_os = "freebsd", any(windows, not(feature = "red")))) {}
    //~^ ERROR: misleading_cfg_in_build_script

    // Should not warn.
    if cfg!(any()) {}
    if cfg!(all()) {}
    if cfg!(feature = "blue") {}
    if cfg!(bob) {}
}
