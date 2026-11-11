use std::env;

#[cfg(miri)]
compile_error!("`miri` cfg should not be set in build script");

fn not_in_miri() -> i32 {
    // Inline assembly definitely does not work in Miri.
    let mut dummy = 42;
    unsafe {
        std::arch::asm!("/* {} */", in(reg) &mut dummy);
    }
    return dummy;
}

fn main() {
    not_in_miri();
    // Cargo calls `miri --print=cfg` to populate the `CARGO_CFG_*` env vars.
    // Make sure that the "miri" flag is set.
    assert!(env::var_os("CARGO_CFG_MIRI").is_some(), "cargo failed to tell us about `--cfg miri`");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MIRITESTVAR");
    println!("cargo:rustc-env=MIRITESTVAR=testval");

    // Test that autocfg works. This invokes RUSTC.
    let a = autocfg::new();
    assert!(a.probe_sysroot_crate("std"));
    assert!(!a.probe_sysroot_crate("doesnotexist"));
    assert!(a.probe_rustc_version(1, 0));
    assert!(!a.probe_rustc_version(2, 0));
    assert!(a.probe_type("i128"));
    assert!(!a.probe_type("doesnotexist"));
    assert!(a.probe_trait("Send"));
    assert!(!a.probe_trait("doesnotexist"));
    assert!(a.probe_path("std::num"));
    assert!(!a.probe_path("doesnotexist"));
    assert!(a.probe_constant("i32::MAX"));
    assert!(!a.probe_constant("doesnotexist"));
    assert!(a.probe_expression("Box::new(0)"));
    assert!(!a.probe_expression("doesnotexist"));
}
