fn main() {
    // Don't rebuild cargo-bsan when nothing changed.
    println!("cargo:rerun-if-changed=build.rs");
    // gather version info
    rustc_tools_util::setup_version_info!();
}
