//@ only-elf
//@ ignore-cross-compile: Runs a binary.
//@ needs-dynamic-linking
// FIXME(raw_dylib_elf): Debug the failures on other targets.
//@ only-gnu
//@ only-x86_64

//! Ensure ELF raw-dylib is able to link against the verbatim versioned library
//! without it being present, and then be executed against this library.

use run_make_support::{build_native_dynamic_lib, cwd, diff, rfs, run, rustc};

fn main() {
    // We compile the binary without having the library present.
    // We also set the rpath to the current directory so we can pick up the library at runtime.
    rustc()
        .crate_type("bin")
        .input("main.rs")
        .arg(&format!("-Wl,-rpath={}", cwd().display()))
        .run();

    // Now, *after* building the binary, we build the library...
    build_native_dynamic_lib("library");
    // ... rename it to have the versioned library name...
    rfs::rename("liblibrary.so", "liblibrary.so.1");

    // ... and run with this library, ensuring it was linked correctly at runtime.
    let output = run("main").stdout_utf8();

    diff().expected_file("output.txt").actual_text("actual", output).run();
}
