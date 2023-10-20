use std::env; // Import 'env' module.

fn main() {
    let host = env::var("HOST").unwrap(); // Get 'HOST' environment variable.

    println!("cargo:rerun-if-changed=build.rs"); // Rerun on 'build.rs' change.
    println!("cargo:rustc-env=BUILD_TRIPLE={host}"); // Set 'BUILD_TRIPLE' environment variable.
}
