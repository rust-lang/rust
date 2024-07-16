// When the TMP or TMPDIR variable is set to an invalid or non-existing directory,
// this used to cause an internal compiler error (ICE). After the addition of proper
// error handling in #28430, this test checks that the expected message is printed.
// See https://github.com/rust-lang/rust/issues/14698

use run_make_support::rustc;

// NOTE: This is not a UI test despite its simplicity, as the error message contains a path
// with some variability that is difficult to normalize

fn main() {
    rustc()
        .input("foo.rs")
        .env("TMP", "fake")
        .env("TMPDIR", "fake")
        .run_fail()
        .assert_stderr_contains("couldn't create a temp dir");
}
