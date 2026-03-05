fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(enable_unstable_features)");

    // Miri testing uses unstable features, so always enable that for its sysroot.
    // Otherwise, only enable unstable if rustc looks like a nightly or dev build.
    let enable_unstable_features = std::env::var("MIRI_CALLED_FROM_SETUP").is_ok() || {
        let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
        let version = std::process::Command::new(rustc).arg("-vV").output().unwrap();
        let stdout = String::from_utf8(version.stdout).unwrap();
        stdout.contains("nightly") || stdout.contains("dev")
    };

    if enable_unstable_features {
        println!("cargo:rustc-cfg=enable_unstable_features");
    }
}
