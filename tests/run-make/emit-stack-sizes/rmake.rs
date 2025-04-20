// Running rustc with the -Z emit-stack-sizes
// flag enables diagnostics to seek stack overflows
// at compile time. This test compiles a rust file
// with this flag, then checks that the output object
// file contains the section "stack_sizes", where
// this diagnostics information should be located.
// See https://github.com/rust-lang/rust/pull/51946

//@ only-elf

use run_make_support::{llvm_readobj, rustc, target};

fn main() {
    rustc()
        .target(target())
        .opt_level("3")
        .arg("-Zemit-stack-sizes")
        .emit("obj")
        .input("foo.rs")
        .run();
    llvm_readobj()
        .arg("--section-headers")
        .input("foo.o")
        .run()
        .assert_stdout_contains(".stack_sizes");
}
