fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CFG_RELEASE");
    println!("cargo:rerun-if-env-changed=CFG_VERSION");
    println!("cargo:rerun-if-env-changed=CFG_VER_DATE");
    println!("cargo:rerun-if-env-changed=CFG_VER_HASH");
}
