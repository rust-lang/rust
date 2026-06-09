//@ only-elf
//@ ignore-cross-compile: Runs a binary.
//@ needs-dynamic-linking
// FIXME(raw_dylib_elf): Debug the failures on other targets.
//@ only-gnu
//@ only-x86_64

//! Ensure ELF raw-dylib is able to link against a non-existent verbatim absolute path
//! by embedding the absolute path in the DT_SONAME and passing a different path for
//! the linker for the stub.

use run_make_support::{build_native_dynamic_lib, cwd, diff, rfs, run, rustc};

fn main() {
    // We compile the binary without having the library present.
    // The verbatim library name is an absolute path.
    rustc().crate_type("bin").input("main.rs").run();

    // FIXME(raw_dylib_elf): Read the NEEDED of the library to ensure it's the absolute path.
}
