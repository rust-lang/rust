//! Make sure that cross-language LTO works on riscv targets,
//! which requires extra abi metadata to be emitted.
//@ needs-matching-clang
//@ needs-llvm-components riscv
extern crate run_make_support;

use run_make_support::{bin_name, rustc, tmp_dir};
use std::{
    env,
    path::PathBuf,
    process::{Command, Output},
    str,
};

fn handle_failed_output(output: Output) {
    eprintln!("output status: `{}`", output.status);
    eprintln!("=== STDOUT ===\n{}\n\n", String::from_utf8(output.stdout).unwrap());
    eprintln!("=== STDERR ===\n{}\n\n", String::from_utf8(output.stderr).unwrap());
    std::process::exit(1)
}

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
    let clang = env::var("CLANG").unwrap();
    let mut cmd = Command::new(clang);
    let executable = tmp_dir().join("riscv-xlto");
    cmd.arg("-target")
        .arg(clang_target)
        .arg(format!("-march={carch}"))
        .arg(format!("-flto=thin"))
        .arg(format!("-fuse-ld=lld"))
        .arg("-nostdlib")
        .arg("-o")
        .arg(&executable)
        .arg("cstart.c")
        .arg(tmp_dir().join("libriscv_xlto.rlib"));
    eprintln!("{cmd:?}");
    let output = cmd.output().unwrap();
    if !output.status.success() {
        handle_failed_output(output);
    }
    // Check that the built binary has correct float abi
    let llvm_readobj =
        PathBuf::from(env::var("LLVM_BIN_DIR").unwrap()).join(bin_name("llvm-readobj"));
    let mut cmd = Command::new(llvm_readobj);
    cmd.arg("--file-header").arg(executable);
    eprintln!("{cmd:?}");
    let output = cmd.output().unwrap();
    if output.status.success() {
        assert!(
            !(is_double_float
                ^ dbg!(str::from_utf8(&output.stdout).unwrap())
                    .contains("EF_RISCV_FLOAT_ABI_DOUBLE"))
        )
    } else {
        handle_failed_output(output);
    }
}

fn main() {
    check_target("riscv64gc-unknown-linux-gnu", "riscv64-linux-gnu", "rv64gc", true);
    check_target("riscv64imac-unknown-none-elf", "riscv64-unknown-elf", "rv64imac", false);
    check_target("riscv32imac-unknown-none-elf", "riscv32-unknown-elf", "rv32imac", false);
    check_target("riscv32gc-unknown-linux-gnu", "riscv32-linux-gnu", "rv32gc", true);
}
