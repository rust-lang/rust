use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(assert_no_panic)");

    println!("cargo:rustc-check-cfg=cfg(feature, values(\"checked\"))");

    #[allow(unexpected_cfgs)]
    if !cfg!(feature = "checked") {
        let lvl = env::var("OPT_LEVEL").unwrap();
        if lvl != "0" {
            println!("cargo:rustc-cfg=assert_no_panic");
        }
    }

    configure_intrinsics();
    configure_arch();
}

/// Simplify the feature logic for enabling intrinsics so code only needs to use
/// `cfg(intrinsics_enabled)`.
fn configure_intrinsics() {
    println!("cargo:rustc-check-cfg=cfg(intrinsics_enabled)");

    // Disabled by default; `unstable-intrinsics` enables again; `force-soft-floats` overrides
    // to disable.
    if cfg!(feature = "unstable-intrinsics") && !cfg!(feature = "force-soft-floats") {
        println!("cargo:rustc-cfg=intrinsics_enabled");
    }
}

/// Simplify the feature logic for enabling arch-specific features so code only needs to use
/// `cfg(arch_enabled)`.
fn configure_arch() {
    println!("cargo:rustc-check-cfg=cfg(arch_enabled)");

    // Enabled by default via the "arch" feature, `force-soft-floats` overrides to disable.
    if cfg!(feature = "arch") && !cfg!(feature = "force-soft-floats") {
        println!("cargo:rustc-cfg=arch_enabled");
    }
}
