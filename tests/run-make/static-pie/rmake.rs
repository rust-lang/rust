// How to manually run this
// $ ./x.py test --target x86_64-unknown-linux-[musl,gnu] tests/run-make/static-pie

//@ only-x86_64
//@ only-linux
//@ ignore-32bit

use std::process::Command;

use run_make_support::regex::Regex;
use run_make_support::{cmd, llvm_readobj, run_with_args, rustc, target};

// Minimum major versions supporting -static-pie
const GCC_VERSION: u32 = 8;
const CLANG_VERSION: u32 = 9;

// Return `true` if the `compiler` version supports `-static-pie`.
fn ok_compiler_version(compiler: &str) -> bool {
    let (trigger, version_threshold) = match compiler {
        "clang" => ("__clang_major__", CLANG_VERSION),
        "gcc" => ("__GNUC__", GCC_VERSION),
        other => panic!("unexpected compiler '{other}', expected 'clang' or 'gcc'"),
    };

    if Command::new(compiler).spawn().is_err() {
        eprintln!("No {compiler} version detected");
        return false;
    }

    let compiler_output =
        cmd(compiler).stdin_buf(trigger).arg("-").arg("-E").arg("-x").arg("c").run().stdout_utf8();
    let re = Regex::new(r"(?m)^(\d+)").unwrap();
    let version: u32 =
        re.captures(&compiler_output).unwrap().get(1).unwrap().as_str().parse().unwrap();

    if version >= version_threshold {
        eprintln!("{compiler} supports -static-pie");
        true
    } else {
        eprintln!("{compiler} too old to support -static-pie, skipping test");
        false
    }
}

fn test(compiler: &str) {
    if !ok_compiler_version(compiler) {
        return;
    }

    rustc()
        .input("test-aslr.rs")
        .linker(compiler)
        .arg("-Clinker-flavor=gcc")
        .arg("-Ctarget-feature=+crt-static")
        .run();

    llvm_readobj()
        .symbols()
        .input("test-aslr")
        .run()
        .assert_stdout_not_contains("INTERP")
        .assert_stdout_contains("DYNAMIC");

    run_with_args("test-aslr", &["--test-aslr"]);
}

fn main() {
    test("clang");
    test("gcc");
}
