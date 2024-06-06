//! Make sure that cross-language LTO works on riscv targets,
//! which requires extra `target-abi` metadata to be emitted.
//@ needs-matching-clang
//@ needs-llvm-components riscv

use run_make_support::{bin_name, clang, llvm_readobj, rustc};
use std::{
    env,
    path::PathBuf,
    process::{Command, Output},
    str,
};

fn check_target(target: &str, clang_target: &str, carch: &str, is_double_float: bool) {
    eprintln!("Checking target {target}");
    // Rust part
    rustc()
        .input("riscv-xlto.rs")
        .crate_type("rlib")
        .target(target)
        .panic("abort")
        .linker_plugin_lto("on")
        .run();
    // C part
    clang()
        .target(clang_target)
        .arch(carch)
        .lto("thin")
        .use_ld("lld")
        .no_stdlib()
        .out_exe("riscv-xlto")
        .input("cstart.c")
        .input("libriscv_xlto.rlib")
        .run();

    // Check that the built binary has correct float abi
    let executable = bin_name("riscv-xlto");
    let output = llvm_readobj().input(&executable).file_header().run();
    let stdout = String::from_utf8_lossy(&output.stdout);
    eprintln!("obj:\n{}", stdout);

    assert!(!(is_double_float ^ stdout.contains("EF_RISCV_FLOAT_ABI_DOUBLE")));
}

fn main() {
    check_target("riscv64gc-unknown-linux-gnu", "riscv64-linux-gnu", "rv64gc", true);
    check_target("riscv64imac-unknown-none-elf", "riscv64-unknown-elf", "rv64imac", false);
    check_target("riscv32imac-unknown-none-elf", "riscv32-unknown-elf", "rv32imac", false);
    check_target("riscv32gc-unknown-linux-gnu", "riscv32-linux-gnu", "rv32gc", true);
}
