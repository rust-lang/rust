use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn clang_path() -> PathBuf {
    if let Ok(d) = env::var("LLVM_BIN_DIR") {
        let clang = Path::new(d.trim_end_matches('/')).join("clang");
        if clang.exists() {
            return clang;
        }
    }
    if let Ok(clang) = env::var("CLANG") {
        let clang = Path::new(clang.trim_end_matches('/'));
        if clang.exists() {
            return clang.to_path_buf();
        }
    }
    PathBuf::from("clang")
}

fn llvm_ar_path() -> PathBuf {
    if let Ok(d) = env::var("LLVM_BIN_DIR") {
        let llvm_ar = Path::new(d.trim_end_matches('/')).join("llvm-ar");
        if llvm_ar.exists() {
            return llvm_ar;
        }
    }
    if let Ok(clang) = env::var("CLANG") {
        let clang = Path::new(&clang);
        if let Some(clang_dir) = clang.parent() {
            let llvm_ar = clang_dir.join("llvm-ar");
            if llvm_ar.exists() {
                return llvm_ar;
            }
        }
    }
    PathBuf::from("llvm-ar")
}

fn build_foo_static_lib(extra_flags: &[&str]) {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR");
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let c_src = Path::new(&manifest_dir).join("src/foo.c");
    let bc_path = Path::new(&out_dir).join("foo.bc");
    let a_path = Path::new(&out_dir).join("libfoo.a");

    let clang = clang_path();
    let llvm_ar = llvm_ar_path();

    let mut clang_args = vec!["-Wall", "-flto=thin", "-fsanitize=cfi"];
    clang_args.extend_from_slice(extra_flags);
    clang_args.extend_from_slice(&["-fvisibility=hidden", "-c", "-emit-llvm", "-o"]);

    let st = Command::new(&clang)
        .args(&clang_args)
        .arg(&bc_path)
        .arg(&c_src)
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn `{}`: {e}", clang.display()));
    assert!(st.success(), "`{}` failed with {st}", clang.display());

    let st = Command::new(&llvm_ar)
        .args(["rcs", a_path.to_str().unwrap(), bc_path.to_str().unwrap()])
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn `{}`: {e}", llvm_ar.display()));
    assert!(st.success(), "`{}` failed with {st}", llvm_ar.display());

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=foo");
    println!("cargo:rerun-if-changed={}", c_src.display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../shared_build_rs.rs");
}
