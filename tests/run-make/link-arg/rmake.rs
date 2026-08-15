//@ needs-target-std
//
// In 2016, the rustc flag "-C link-arg" was introduced - it can be repeatedly used
// to add single arguments to the linker. This test passes 2 arguments to the linker using it,
// then checks that the compiler's output contains the arguments passed to it.
// This ensures that the compiler successfully parses this flag.
// See https://github.com/rust-lang/rust/pull/36574

use run_make_support::rustc;

fn main() {
    // We are only checking for the output of --print=link-args,
    // rustc failing or succeeding does not matter.
    let out = rustc()
        .input("empty.rs")
        .link_arg("-lfoo")
        .link_arg("-lbar")
        .print("link-args")
        .run_unchecked();
    out.assert_stdout_contains("lfoo");
    out.assert_stdout_contains("lbar");
    assert!(out.stdout_utf8().ends_with('\n'));
}
