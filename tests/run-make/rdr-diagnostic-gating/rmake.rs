// Check that diagnostic replay is gated by the spans hash when using -Z stable-crate-hash.
//@ ignore-cross-compile
// Reason: uses incremental directory tied to host toolchain paths

use run_make_support::{rfs, run_in_tmpdir, rustc};

fn main() {
    run_in_tmpdir(|| {
        let source_v1 = "fn main() { let unused = 1; }\n";
        rfs::write("main.rs", source_v1);

        let output1 = rustc()
            .input("main.rs")
            .incremental("incr")
            .arg("-Zstable-crate-hash")
            .arg("-Zincremental-ignore-spans")
            .run();
        output1.assert_stderr_contains("unused variable");

        let output2 = rustc()
            .input("main.rs")
            .incremental("incr")
            .arg("-Zstable-crate-hash")
            .arg("-Zincremental-ignore-spans")
            .run();
        output2.assert_stderr_contains("unused variable");

        let source_v2 = "// span-only change\nfn main() { let unused = 1; }\n";
        rfs::write("main.rs", source_v2);

        let output3 = rustc()
            .input("main.rs")
            .incremental("incr")
            .arg("-Zstable-crate-hash")
            .arg("-Zincremental-ignore-spans")
            .run();
        output3.assert_stderr_not_contains("unused variable");
    });
}
