// How to manually run this
// $ ./x.py test --target x86_64-unknown-linux-[musl,gnu] tests/run-make/static-pie

//@ only-x86_64
//@ only-linux
//@ ignore-32bit

use std::process::Command;

use run_make_support::llvm_readobj;
use run_make_support::rustc;
use run_make_support::{cmd, run_with_args, target};

fn ok_compiler_version(compiler: &str) -> bool {
    let check_file = format!("check_{compiler}_version.sh");

    Command::new(check_file).status().is_ok_and(|status| status.success())
}

fn test(compiler: &str) {
    if !ok_compiler_version(compiler) {
        return;
    }

    rustc()
        .input("test-aslr.rs")
        .target(&target())
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
