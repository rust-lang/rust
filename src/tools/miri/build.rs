fn main() {
    // Don't rebuild miri when nothing changed.
    println!("cargo:rerun-if-changed=build.rs");
    // Re-export the TARGET environment variable so it can be accessed by miri. Needed to know the
    // "host" triple inside Miri.
    let target = std::env::var("TARGET").unwrap();
    println!("cargo:rustc-env=TARGET={target}");
    // Allow some cfgs.
    println!("cargo::rustc-check-cfg=cfg(bootstrap)");
}
