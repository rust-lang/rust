// This tests the different -Crelro-level values, and makes sure that they work properly.

//@ only-linux
//@ ignore-cross-compile

use run_make_support::{llvm_readobj, rustc};

fn compile(relro_level: &str) {
    rustc().arg(format!("-Crelro-level={relro_level}")).input("hello.rs").run();
}

fn main() {
    // Ensure that binaries built with the full relro level links them with both
    // RELRO and BIND_NOW for doing eager symbol resolving.

    compile("full");
    llvm_readobj().program_headers().input("hello").run().assert_stdout_contains("GNU_RELRO");
    llvm_readobj().dynamic_table().input("hello").run().assert_stdout_contains("BIND_NOW");

    compile("partial");
    llvm_readobj().program_headers().input("hello").run().assert_stdout_contains("GNU_RELRO");

    // Ensure that we're *not* built with RELRO when setting it to off.  We do
    // not want to check for BIND_NOW however, as the linker might have that
    // enabled by default.
    compile("off");
    llvm_readobj().program_headers().input("hello").run().assert_stdout_not_contains("GNU_RELRO");
}
