// Invalid paths passed to rustc used to cause internal compilation errors
// alongside an obscure error message. This was turned into a standard error,
// and this test checks that the cleaner error message is printed instead.
// See https://github.com/rust-lang/rust/issues/13517

use run_make_support::rustc;

// NOTE: This cannot be a UI test due to the --out-dir flag, which is
// already present by default in UI testing.

fn main() {
    let out = rustc().input("foo.rs").emit("dep-info").out_dir("foo/bar/baz").run_fail();
    // The error message should be informative.
    out.assert_stderr_contains("error writing dependencies");
    // The filename should appear.
    out.assert_stderr_contains("baz");
}
