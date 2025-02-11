//! Test the semantics of `cfg(os_version_min)` while setting deployment target.

//@ only-macos
//@ revisions: no_env env_low env_mid env_high
//@[no_env] unset-rustc-env:MACOSX_DEPLOYMENT_TARGET
//@[env_low] rustc-env:MACOSX_DEPLOYMENT_TARGET=10.14
//@[env_mid] rustc-env:MACOSX_DEPLOYMENT_TARGET=14.0
//@[env_high] rustc-env:MACOSX_DEPLOYMENT_TARGET=17.0
//@ run-pass

#![feature(cfg_os_version_min)]

fn main() {
    assert_eq!(cfg!(os_version_min("macos", "14.0")), cfg!(any(env_mid, env_high)));

    // Aarch64 minimum is macOS 11.0, even if a lower env is requested.
    assert_eq!(
        cfg!(os_version_min("macos", "11.0")),
        cfg!(any(env_mid, env_high, target_arch = "aarch64"))
    );
}
