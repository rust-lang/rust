use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Automatically detect if thumb-mode is an available feature by looking at
    // the prefix of the target. Currently, the thumb-mode target feature is
    // only set automatically in nightly builds, so we must do the manual
    // feature detect here.
    //
    // "armv7-linux-androideabi" is a special case that has thumb-mode enabled,
    // but does not start with the "thumb" prefix.
    if env::var("TARGET").map_or(false, |t| {
        t.starts_with("thumb") || t == "armv7-linux-androideabi"
    }) {
        println!("cargo:rustc-cfg=target_feature=\"thumb-mode\"");
    }
}
