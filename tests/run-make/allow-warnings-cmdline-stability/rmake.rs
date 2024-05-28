// Test that `-Awarnings` suppresses warnings for unstable APIs.

use run_make_support::{assert_not_contains, rustc};

fn main() {
    rustc().input("bar.rs").run();
    let output = rustc().input("foo.rs").arg("-Awarnings").run();

    assert_not_contains(&String::from_utf8(output.stdout).unwrap(), "warning");
    assert_not_contains(&String::from_utf8(output.stderr).unwrap(), "warning");
}
