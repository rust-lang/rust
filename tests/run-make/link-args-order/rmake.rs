// Passing linker arguments to the compiler used to be lost or reordered in a messy way
// as they were passed further to the linker. This was fixed in #70665, and this test
// checks that linker arguments remain intact and in the order they were originally passed in.
// See https://github.com/rust-lang/rust/pull/70665

use run_make_support::rustc;

fn main() {
    rustc()
        .input("empty.rs")
        .linker_flavor("ld")
        .link_arg("a")
        .link_args("\"b c\"")
        .link_args("\"d e\"")
        .link_arg("f")
        .run_fail()
        .assert_stderr_contains("\"a\" \"b\" \"c\" \"d\" \"e\" \"f\"");
    rustc()
        .input("empty.rs")
        .linker_flavor("ld")
        .pre_link_arg("a")
        .pre_link_args("\"b c\"")
        .pre_link_args("\"d e\"")
        .pre_link_arg("f")
        .run_fail()
        .assert_stderr_contains("\"a\" \"b\" \"c\" \"d\" \"e\" \"f\"");
}
