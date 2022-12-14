fn main() {
    // Don't rebuild miri when nothing changed.
    println!("cargo:rerun-if-changed=build.rs");
    // Re-export the TARGET environment variable so it can
    // be accessed by miri.
    let target = std::env::var("TARGET").unwrap();
    println!("cargo:rustc-env=TARGET={target}");
}
