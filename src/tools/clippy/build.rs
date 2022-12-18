fn main() {
    // Forward the profile to the main compilation
    println!("cargo:rustc-env=PROFILE={}", std::env::var("PROFILE").unwrap());
    // Don't rebuild even if nothing changed
    println!("cargo:rerun-if-changed=build.rs");
    rustc_tools_util::setup_version_info!();
}
