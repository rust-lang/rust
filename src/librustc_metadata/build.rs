fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CFG_VERSION");
    println!("cargo:rerun-if-env-changed=CFG_VIRTUAL_RUST_SOURCE_BASE_DIR");
}
