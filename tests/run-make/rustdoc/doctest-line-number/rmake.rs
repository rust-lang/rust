// When a docblock contains attributes or code comments which are not part
// of the final docstring, they are not taken into account when computing
// lines offset. This test ensures that.
// Regression test for <https://github.com/rust-lang/rust/issues/164685>.

//@ ignore-cross-compile

use run_make_support::rustdoc;

fn main() {
    let out = rustdoc().input("foo.rs").arg("--test").run().stdout_utf8();

    assert!(out.contains("running 3 tests"));
    assert!(out.contains("test foo.rs - (line 2) ... ignored"));
    assert!(out.contains("test foo.rs - (line 8) ... ignored"));
    assert!(out.contains("test foo.rs - (line 14) ... ok"));
}
