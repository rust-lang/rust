// This test checks that the embed bitcode in elf created with
// lto-embed-bitcode=optimized is a valid llvm bitcode module.
// Otherwise, the `test.bc` file will cause an error when
// `llvm-dis` attempts to disassemble it.
// See https://github.com/rust-lang/rust/issues/84395

//@ needs-force-clang-based-tests
// NOTE(#126180): This test only runs on `x86_64-gnu-debug`, because that CI job sets
// RUSTBUILD_FORCE_CLANG_BASED_TESTS and only runs tests which contain "clang" in their
// name.

use run_make_support::llvm::llvm_bin_dir;
use run_make_support::{cmd, env_var, rustc};

fn main() {
    rustc()
        .input("test.rs")
        .link_arg("-fuse-ld=lld")
        .arg("-Clinker-plugin-lto")
        .linker(&env_var("CLANG"))
        .link_arg("-Wl,--plugin-opt=-lto-embed-bitcode=optimized")
        .arg("-Zemit-thin-lto=no")
        .run();
    cmd(llvm_bin_dir().join("llvm-objcopy"))
        .arg("--dump-section")
        .arg(".llvmbc=test.bc")
        .arg("test")
        .run();
    cmd(llvm_bin_dir().join("llvm-dis")).arg("test.bc").run();
}
