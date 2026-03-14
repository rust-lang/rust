// This test ensures that the `misleading_cfg_in_build_script` is not emitted if not
// in a cargo `build.rs` script.
//
//@ no-auto-check-cfg
//@ check-pass

#![deny(misleading_cfg_in_build_script)]
#![allow(dead_code)]

#[cfg(windows)]
fn unused_windows_fn() {}
#[cfg(not(windows))]
fn unused_not_windows_fn() {}
#[cfg(any(windows, feature = "yellow", unix))]
fn pink() {}

#[cfg(feature = "green")]
fn pink() {}
#[cfg(bob)]
fn bob() {}

fn main() {
    if cfg!(windows) {}
    if cfg!(not(windows)) {}
    if cfg!(target_os = "freebsd") {}
    if cfg!(any(target_os = "freebsd", windows)) {}
    if cfg!(not(any(target_os = "freebsd", windows))) {}
    if cfg!(all(target_os = "freebsd", windows)) {}
    if cfg!(not(all(target_os = "freebsd", windows))) {}
    if cfg!(all(target_os = "freebsd", any(windows, not(feature = "red")))) {}

    if cfg!(any()) {}
    if cfg!(all()) {}
    if cfg!(feature = "blue") {}
    if cfg!(bob) {}
}
