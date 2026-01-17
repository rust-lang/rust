// Test that panic locations inside async functions are correct
// when using -Z separate-spans.

//@ ignore-cross-compile

use run_make_support::{cmd, run_in_tmpdir, rustc};

fn main() {
    run_in_tmpdir(|| {
        rustc()
            .input("lib.rs")
            .crate_name("rdr_async_lib")
            .crate_type("rlib")
            .edition("2024")
            .arg("-Zseparate-spans")
            .run();

        rustc()
            .input("main.rs")
            .crate_type("bin")
            .extern_("rdr_async_lib", "librdr_async_lib.rlib")
            .edition("2024")
            .arg("-Zseparate-spans")
            .run();

        let output = cmd("./main").arg("async").run_fail();
        let stderr = output.stderr_utf8();
        assert!(
            stderr.contains("lib.rs:6"),
            "async panic should show lib.rs:6, got:\n{stderr}"
        );

        let output = cmd("./main").arg("nested").run_fail();
        let stderr = output.stderr_utf8();
        assert!(
            stderr.contains("lib.rs:11"),
            "nested async panic should show lib.rs:11, got:\n{stderr}"
        );
    });
}
