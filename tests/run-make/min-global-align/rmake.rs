// This test checks that global variables respect the target minimum alignment.
// The three bools `STATIC_BOOL`, `STATIC_MUT_BOOL`, and `CONST_BOOL` all have
// type-alignment of 1, but some targets require greater global alignment.
// See https://github.com/rust-lang/rust/pull/44440

//@ only-linux
// Reason: this test is specific to linux, considering compilation is targeted
// towards linux architectures only.

use run_make_support::{assert_count_is, llvm_components_contain, rfs, rustc};

fn main() {
    // Most targets are happy with default alignment -- take i686 for example.
    if llvm_components_contain("x86") {
        rustc().target("i686-unknown-linux-gnu").emit("llvm-ir").input("min_global_align.rs").run();
        assert_count_is(3, rfs::read_to_string("min_global_align.ll"), "align 1");
    }
    // SystemZ requires even alignment for PC-relative addressing.
    if llvm_components_contain("systemz") {
        rustc()
            .target("s390x-unknown-linux-gnu")
            .emit("llvm-ir")
            .input("min_global_align.rs")
            .run();
        assert_count_is(3, rfs::read_to_string("min_global_align.ll"), "align 2");
    }
}
