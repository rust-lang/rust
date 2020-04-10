use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC_VERIFY_LLVM_IR");

    if env::var_os("RUSTC_VERIFY_LLVM_IR").is_some() {
        println!("cargo:rustc-cfg=always_verify_llvm_ir");
    }
}
