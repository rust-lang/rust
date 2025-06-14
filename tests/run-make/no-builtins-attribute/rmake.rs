//@ needs-target-std
//
// `no_builtins` is an attribute related to LLVM's optimizations. In order to ensure that it has an
// effect on link-time optimizations (LTO), it should be added to function declarations in a crate.
// This test uses the `llvm-filecheck` tool to determine that this attribute is successfully
// being added to these function declarations.
// See https://github.com/rust-lang/rust/pull/113716

use run_make_support::{llvm_filecheck, rfs, rustc};

fn main() {
    rustc().input("no_builtins.rs").emit("link").run();
    rustc().input("main.rs").emit("llvm-ir").run();
    llvm_filecheck().patterns("filecheck.main.txt").stdin_buf(rfs::read("main.ll")).run();
}
