// Regression test for <https://github.com/rust-lang/rust/issues/143981>.

//@ ignore-cross-compile

// NOTE: `sdylib`'s platform support is basically that of `dylib`.
//@ needs-crate-type: dylib

use run_make_support::rustc;

fn main() {
    rustc()
        .input("libr.rs")
        .output("does-not-exist/output")
        .run_fail()
        .assert_exit_code(1)
        .assert_stderr_contains("failed to write file")
        .assert_not_ice();
}
