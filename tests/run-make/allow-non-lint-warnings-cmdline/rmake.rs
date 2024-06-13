// Test that -A warnings makes the 'empty trait list for derive' warning go away.

use run_make_support::rustc;

fn main() {
    let output = rustc().input("foo.rs").arg("-Awarnings").run();
    output.assert_stderr_not_contains("warning");
}
