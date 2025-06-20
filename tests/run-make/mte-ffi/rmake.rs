//! Tests that MTE tags and values stored in the top byte of a pointer (TBI) are preserved across
//! FFI boundaries (C <-> Rust). This test does not require MTE: whilst the test will use MTE if
//! available, if it is not, arbitrary tag bits are set using TBI.

//@ only-aarch64-unknown-linux-gnu
// Reason: this test is only valid for AArch64 with `gcc`. The linker must be explicitly specified
// when cross-compiling, so it is limited to `aarch64-unknown-linux-gnu`.

use run_make_support::{dynamic_lib_name, extra_c_flags, gcc, run, rustc, target};

fn main() {
    run_test("int");
    run_test("float");
    run_test("string");
    run_test("function");
}

fn run_test(variant: &str) {
    let flags = {
        let mut flags = extra_c_flags();
        flags.push("-march=armv8.5-a+memtag");
        flags
    };
    println!("{variant} test...");
    rustc().input(format!("foo_{variant}.rs")).linker("aarch64-linux-gnu-gcc").run();
    gcc()
        .input(format!("bar_{variant}.c"))
        .input(dynamic_lib_name("foo"))
        .out_exe("test")
        .args(&flags)
        .run();
    run("test");
}
