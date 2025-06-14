//@ needs-target-std
// Test that `-Awarnings` suppresses warnings for unstable APIs.

use run_make_support::rustc;

fn main() {
    rustc().input("bar.rs").run();
    rustc()
        .input("foo.rs")
        .arg("-Awarnings")
        .run()
        .assert_stdout_not_contains("warning")
        .assert_stderr_not_contains("warning");
}
