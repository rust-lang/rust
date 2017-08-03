use std::env;

fn main() {
    // Forward the profile to the main compilation
    println!("cargo:rustc-env=PROFILE={}", env::var("PROFILE").unwrap());
}
