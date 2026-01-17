// Test that panic locations in proc-macro-generated code are correct
// when using -Z separate-spans.

//@ ignore-cross-compile

use run_make_support::{cmd, run_in_tmpdir, rustc};

fn main() {
    run_in_tmpdir(|| {
        rustc()
            .input("proc_macro_lib.rs")
            .crate_type("proc-macro")
            .arg("-Zseparate-spans")
            .run();

        rustc()
            .input("main.rs")
            .crate_type("bin")
            .extern_("proc_macro_lib", "libproc_macro_lib.dylib")
            .arg("-Zseparate-spans")
            .run();

        let output = cmd("./main").arg("derive").run_fail();
        let stderr = output.stderr_utf8();
        assert!(
            stderr.contains("panic from derived impl"),
            "should contain panic message, got:\n{stderr}"
        );

        let output = cmd("./main").arg("attr").run_fail();
        let stderr = output.stderr_utf8();
        assert!(
            stderr.contains("panic from attribute macro"),
            "should contain panic message, got:\n{stderr}"
        );
    });
}
