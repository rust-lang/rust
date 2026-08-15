//@ only-elf
//@ ignore-cross-compile: Runs a binary.
//@ needs-dynamic-linking
// FIXME(raw_dylib_elf): Debug the failures on other targets.
//@ only-gnu
//@ only-x86_64

//! Ensure ELF raw-dylib is able to link the binary without having the library present,
//! and then successfully run against the real library.

use run_make_support::{build_native_dynamic_lib, cwd, diff, run, rustc};

fn main() {
    // We compile the binaries without having the library present with different relocation
    // models.
    // We also set the rpath to the current directory so we can pick up the library at runtime.
    let relocation_models = ["static", "pic", "pie"];
    for relocation_model in relocation_models {
        rustc()
            .crate_type("bin")
            .input("main.rs")
            .arg(&format!("-Wl,-rpath={}", cwd().display()))
            .arg(&format!("-Crelocation-model={}", relocation_model))
            .output(&format!("main-{}", relocation_model))
            .run();
    }

    // Now, *after* building the binaries, we build the library...
    build_native_dynamic_lib("library");

    for relocation_model in relocation_models {
        // ... and run with this library, ensuring it was linked correctly at runtime.
        // The output here should be the same regardless of the relocation model.
        let output = run(&format!("main-{}", relocation_model)).stdout_utf8();
        diff().expected_file("output.txt").actual_text("actual", output).run();
    }
}
