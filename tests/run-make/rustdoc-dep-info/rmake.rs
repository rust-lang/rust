// This is a simple smoke test for rustdoc's `--emit dep-info` feature. It prints out
// information about dependencies in a Makefile-compatible format, as a `.d` file.

//@ needs-target-std

use run_make_support::assertion_helpers::assert_contains;
use run_make_support::{path, rfs, rustdoc};

fn main() {
    // We're only emitting dep info, so we shouldn't be running static analysis to
    // figure out that this program is erroneous.
    // Ensure that all kinds of input reading flags end up in dep-info.
    rustdoc()
        .input("lib.rs")
        .arg("-Zunstable-options")
        .arg("--html-before-content=before.html")
        .arg("--markdown-after-content=after.md")
        .arg("--extend-css=extend.css")
        .arg("--theme=theme.css")
        .emit("dep-info")
        .run();

    let content = rfs::read_to_string("foo.d");
    assert_contains(&content, "lib.rs:");
    assert_contains(&content, "foo.rs:");
    assert_contains(&content, "bar.rs:");
    assert_contains(&content, "doc.md:");
    assert_contains(&content, "after.md:");
    assert_contains(&content, "before.html:");
    assert_contains(&content, "extend.css:");
    assert_contains(&content, "theme.css:");

    // Now we check that we can provide a file name to the `dep-info` argument.
    rustdoc().input("lib.rs").arg("-Zunstable-options").emit("dep-info=bla.d").run();
    assert!(path("bla.d").exists());

    // The last emit-type wins. The same behavior as rustc.
    rustdoc()
        .input("lib.rs")
        .arg("-Zunstable-options")
        .emit("dep-info=precedence1.d")
        .emit("dep-info=precedence2.d")
        .emit("dep-info=precedence3.d")
        .run();
    assert!(!path("precedence1.d").exists());
    assert!(!path("precedence2.d").exists());
    assert!(path("precedence3.d").exists());
}
