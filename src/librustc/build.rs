fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CFG_LIBDIR_RELATIVE");
    println!("cargo:rerun-if-env-changed=CFG_COMPILER_HOST_TRIPLE");
}
