//@ needs-target-std
//
// The byte positions in json format error logging used to have a small, difficult
// to predict offset. This was changed to be the top of the file every time in #42973,
// and this test checks that the measurements appearing in the standard error are correct.
// See https://github.com/rust-lang/rust/issues/35164

use run_make_support::rustc;

fn main() {
    rustc()
        .input("main.rs")
        .error_format("json")
        .run()
        .assert_stderr_contains(r#""byte_start":23"#)
        .assert_stderr_contains(r#""byte_end":29"#);
}
