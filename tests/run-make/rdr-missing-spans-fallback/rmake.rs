// Test that missing .spans file is treated as a hard error.
//
// This verifies that when a crate was compiled with `-Z stable-crate-hash` and
// the `.spans` file is later deleted (simulating a corrupted incremental cache),
// the compiler produces a clear error message rather than silently degrading.
//
//@ ignore-cross-compile
// Reason: uses incremental directory tied to host toolchain paths

use run_make_support::{rfs, run_in_tmpdir, rustc};

fn main() {
    run_in_tmpdir(|| {
        // First, create a dependency crate compiled with -Z stable-crate-hash
        // Use --emit=metadata to produce a standalone .rmeta file
        let dep_source = "pub fn dep_fn() -> i32 { 42 }\n";
        rfs::write("dep.rs", dep_source);

        // Compile the dependency with -Z stable-crate-hash and emit both rlib and metadata
        rustc()
            .input("dep.rs")
            .crate_type("rlib")
            .emit("metadata,link")
            .arg("-Zstable-crate-hash")
            .run();

        // Verify .spans file was created alongside .rmeta
        let rmeta_path = std::path::Path::new("libdep.rmeta");
        let spans_path = std::path::Path::new("libdep.spans");
        let rlib_path = std::path::Path::new("libdep.rlib");
        assert!(rmeta_path.exists(), "expected libdep.rmeta to exist");
        assert!(spans_path.exists(), "expected libdep.spans to exist");
        assert!(rlib_path.exists(), "expected libdep.rlib to exist");

        // Delete the .spans file to simulate corrupted cache
        rfs::remove_file(spans_path);

        // Now try to compile a crate that depends on the dependency
        // This should fail with a clear error about missing span data
        let main_source = "extern crate dep; fn main() { dep::dep_fn(); }\n";
        rfs::write("main.rs", main_source);

        let output = rustc().input("main.rs").extern_("dep", "libdep.rlib").run_fail();

        // Verify we get the expected error message
        output.assert_stderr_contains("cannot load span data for crate");
        output.assert_stderr_contains("the incremental compilation cache may be corrupted");
    });
}
