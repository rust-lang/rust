// This test ensures that the file paths displayed in doctests are "beautified",
// meaning they don't contain ".." or "." characters.

use run_make_support::rustdoc;

fn main() {
    rustdoc()
        .input("foo.rs")
        .arg("--test")
        .run_fail()
        .assert_stdout_not_contains("-- ../")
        .assert_stdout_not_contains("-- ./")
        .assert_stdout_contains("-- sub/bar.md ");
}
