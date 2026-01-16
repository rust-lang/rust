// Test that panic locations are correct when using -Z separate-spans.
//
// This verifies that span information is correctly preserved/resolved
// so that panic messages show accurate file:line:col locations.
//
// We test three scenarios:
// 1. Panic in a public function
// 2. Panic in a private function called from a public one
// 3. Panic with format string arguments

//@ ignore-cross-compile

use run_make_support::{cmd, run_in_tmpdir, rustc};

fn main() {
    run_in_tmpdir(|| {
        // Build dependency with -Z separate-spans
        rustc().input("dep.rs").crate_type("rlib").arg("-Zseparate-spans").run();

        // Build main crate linking to the dependency
        rustc()
            .input("main.rs")
            .crate_type("bin")
            .extern_("dep", "libdep.rlib")
            .arg("-Zseparate-spans")
            .run();

        // Test 1: Public function panic location
        // The panic should show dep.rs:13
        let output = cmd("./main").arg("public").run_fail();
        let stderr = output.stderr_utf8();
        assert!(
            stderr.contains("dep.rs:13"),
            "Panic in public function should show dep.rs:13, got:\n{}",
            stderr
        );
        assert!(stderr.contains("intentional panic for testing"), "Should contain panic message");

        // Test 2: Private function panic location
        // The panic should show dep.rs:27
        let output = cmd("./main").arg("private").run_fail();
        let stderr = output.stderr_utf8();
        assert!(
            stderr.contains("dep.rs:27"),
            "Panic in private function should show dep.rs:27, got:\n{}",
            stderr
        );
        assert!(stderr.contains("panic from private function"), "Should contain panic message");

        // Test 3: Format string panic location
        // The panic should show dep.rs:36
        let output = cmd("./main").arg("format").run_fail();
        let stderr = output.stderr_utf8();
        assert!(
            stderr.contains("dep.rs:36"),
            "Panic with format should show dep.rs:36, got:\n{}",
            stderr
        );
        assert!(stderr.contains("invalid value: -1"), "Should contain formatted panic message");

        println!("All panic location tests passed!");
    });
}
