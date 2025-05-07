//! Smoke test to make sure the embed bitcode in elf created with
//! `--plugin-opt=-lto-embed-bitcode=optimized` is valid llvm BC module.
//!
//! See <https://github.com/rust-lang/rust/issues/84395> where passing
//! `-lto-embed-bitcode=optimized` to lld when linking rust code via `linker-plugin-lto` doesn't
//! produce the expected result.
//!
//! See PR <https://github.com/rust-lang/rust/pull/98162> which initially introduced this test.

//@ needs-force-clang-based-tests

use run_make_support::{env_var, llvm_dis, llvm_objcopy, rustc};

fn main() {
    rustc()
        .input("test.rs")
        .arg("-Clink-arg=-fuse-ld=lld")
        .arg("-Clinker-plugin-lto")
        .arg(format!("-Clinker={}", env_var("CLANG")))
        .arg("-Clink-arg=-Wl,--plugin-opt=-lto-embed-bitcode=optimized")
        .arg("-Zemit-thin-lto=no")
        .run();

    llvm_objcopy().dump_section(".llvmbc", "test.bc").arg("test").run();

    llvm_dis().arg("test.bc").run();
}
