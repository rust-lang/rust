fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CFG_RELEASE");
    println!("cargo:rerun-if-env-changed=CFG_RELEASE_CHANNEL");
    println!("cargo:rerun-if-env-changed=CFG_VERSION");
    println!("cargo:rerun-if-env-changed=CFG_VER_DATE");
    println!("cargo:rerun-if-env-changed=CFG_VER_HASH");
    println!("cargo:rerun-if-env-changed=CFG_PREFIX");
    println!("cargo:rerun-if-env-changed=CFG_VIRTUAL_RUST_SOURCE_BASE_DIR");
    println!("cargo:rerun-if-env-changed=CFG_COMPILER_HOST_TRIPLE");
    println!("cargo:rerun-if-env-changed=CFG_LIBDIR_RELATIVE");
}
