//@ needs-target-std
//
// When the TMP (on Windows) or TMPDIR (on Unix) variable is set to an invalid
// or non-existing directory, this used to cause an internal compiler error (ICE). After the
// addition of proper error handling in #28430, this test checks that the expected message is
// printed.
// See https://github.com/rust-lang/rust/issues/14698

use run_make_support::{is_windows, rustc};

// NOTE: This is not a UI test despite its simplicity, as the error message contains a path
// with some variability that is difficult to normalize

fn main() {
    let mut rustc = rustc();
    if is_windows() {
        rustc.env("TMP", "fake");
    } else {
        rustc.env("TMPDIR", "fake");
    }
    rustc.input("foo.rs").run_fail().assert_stderr_contains("couldn't create a temp dir");
}
