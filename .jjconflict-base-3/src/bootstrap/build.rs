use std::env;

fn main() {
    // this is needed because `HOST` is only available to build scripts.
    let host = env::var("HOST").unwrap();
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-env=BUILD_TRIPLE={host}");
}
