fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CFG_RELEASE_CHANNEL");
    println!("cargo:rerun-if-env-changed=CFG_DISABLE_UNSTABLE_FEATURES");
}
