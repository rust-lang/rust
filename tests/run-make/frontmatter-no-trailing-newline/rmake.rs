// Regression test for issue #151882
// See https://github.com/rust-lang/rust/issues/151882

//@ only-nightly
//@ needs-target-std

use run_make_support::{rfs, rustc};

fn main() {
    rfs::write("test.rs", b"----");

    // Ensure rustc does not ICE when parsing a file with frontmatter syntax
    // that has no trailing newline
    rustc()
        .input("test.rs")
        .run_fail()
        .assert_stderr_contains("invalid infostring for frontmatter")
        .assert_stderr_not_contains("unexpectedly panicked");
}
