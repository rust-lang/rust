// When the metadata format changes, old libraries used to cause librustc to abort
// when reading their metadata. The error message for this scenario was unhelpful at best.
// A better error message was implemented in #12645, and this test checks that it is the
// one appearing in stderr in this scenario.
// See https://github.com/rust-lang/rust/pull/12645

use run_make_support::{llvm_ar, rfs, rustc};

fn main() {
    rfs::create_file("lib.rmeta");
    llvm_ar().obj_to_ar().output_input("libfoo-ffffffff-1.0.rlib", "lib.rmeta").run();
    rustc().input("foo.rs").run_fail().assert_stderr_contains("found invalid metadata");
}
