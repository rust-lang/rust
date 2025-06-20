// Running rustc with the -Z emit-stack-sizes
// flag enables diagnostics to seek stack overflows
// at compile time. This test compiles a rust file
// with this flag, then checks that the output object
// file contains the section "stack_sizes", where
// this diagnostics information should be located.
// See https://github.com/rust-lang/rust/pull/51946

//@ needs-target-std
//@ only-elf
// Reason: this feature only works when the output object format is ELF.
// This won't be the case on Windows/OSX - for example, OSX produces a Mach-O binary.

use run_make_support::{llvm_readobj, rustc};

fn main() {
    rustc().opt_level("3").arg("-Zemit-stack-sizes").emit("obj").input("foo.rs").run();
    llvm_readobj()
        .arg("--section-headers")
        .input("foo.o")
        .run()
        .assert_stdout_contains(".stack_sizes");
}
