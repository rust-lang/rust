//! Test low and high version with `cfg(os_version_min)`.
//@ run-pass
#![feature(cfg_os_version_min)]

fn main() {
    // Always available on macOS
    assert_eq!(cfg!(os_version_min("macos", "10.0")), cfg!(target_os = "macos"));
    //~^ WARNING: version is set unnecessarily low

    // Never available
    assert!(cfg!(not(os_version_min("macos", "9999.99.99"))));
}
