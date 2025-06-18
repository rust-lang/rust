//@ needs-target-std
//
// Emitting dep-info alongside metadata would present subtle discrepancies
// in the output file, such as the filename transforming underscores_ into hyphens-.
// After the fix in #114750, this test checks that the emitted files are identical
// to the expected output.
// See https://github.com/rust-lang/rust/issues/68839

use run_make_support::{diff, rustc};

fn main() {
    rustc()
        .emit("metadata,dep-info")
        .crate_type("lib")
        .input("dash-separated.rs")
        .extra_filename("_something-extra")
        .run();
    diff()
        .expected_file("dash-separated_something-extra.expected.d")
        .actual_file("dash-separated_something-extra.d")
        .run();
}
