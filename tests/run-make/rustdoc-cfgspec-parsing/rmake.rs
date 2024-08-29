// A rustdoc bug caused the `feature=bar` syntax for the cfg flag to be interpreted
// wrongly, with `feature=bar` instead of just `bar` being understood as the feature name.
// After this was fixed in #22135, this test checks that this bug does not make a resurgence.
// See https://github.com/rust-lang/rust/issues/22131

//@ ignore-cross-compile
// Reason: rustdoc fails to find the "foo" crate

use run_make_support::{cwd, rustc, rustdoc};

fn main() {
    rustc().cfg(r#"feature="bar""#).crate_type("lib").input("foo.rs").run();
    rustdoc()
        .arg("--test")
        .arg("--cfg")
        .arg(r#"feature="bar""#)
        .library_search_path(cwd())
        .input("foo.rs")
        .run()
        .assert_stdout_contains("foo.rs - foo (line 1) ... ok");
}
