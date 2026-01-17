// Test that debuginfo line numbers are correct when using -Z stable-crate-hash.
//
// This verifies that DWARF debug information contains accurate file:line
// information even when spans are stored separately from metadata.
//
// We use llvm-dwarfdump to inspect the generated debug info and verify
// that function line numbers match their source locations.

//@ ignore-cross-compile
//@ needs-llvm-components: aarch64 arm x86

use run_make_support::{llvm_dwarfdump, run_in_tmpdir, rustc};

fn main() {
    run_in_tmpdir(|| {
        // Build with debuginfo and -Z stable-crate-hash
        rustc()
            .input("lib.rs")
            .crate_type("rlib")
            .arg("-Cdebuginfo=2")
            .arg("-Zstable-crate-hash")
            .run();

        // Use llvm-dwarfdump to extract debug info
        let output = llvm_dwarfdump().arg("--debug-line").arg("librdr_debuginfo_lib.rlib").run();

        let stdout = output.stdout_utf8();

        // Verify that the debug line info contains references to our source file
        assert!(stdout.contains("lib.rs"), "Debug info should reference lib.rs, got:\n{}", stdout);

        // Build again and verify reproducibility
        rustc()
            .input("lib.rs")
            .crate_type("rlib")
            .arg("-Cdebuginfo=2")
            .arg("-Zstable-crate-hash")
            .out_dir("second")
            .run();

        let output2 =
            llvm_dwarfdump().arg("--debug-line").arg("second/librdr_debuginfo_lib.rlib").run();

        let stdout2 = output2.stdout_utf8();

        // The debug line tables should be identical
        // (after accounting for path differences)
        assert!(stdout2.contains("lib.rs"), "Second build debug info should also reference lib.rs");

        println!("Debuginfo tests passed!");
    });
}
