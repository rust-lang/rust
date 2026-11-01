//@ needs-target-std
//
// A flag named dump-mono-stats was added to the compiler in 2022, which
// collects stats on instantiation of items and their associated costs.
// This test checks that the output stat file exists, and that it contains
// a specific expected string.
// See https://github.com/rust-lang/rust/pull/105481

use run_make_support::{cwd, rfs, rustc};

fn main() {
    rustc()
        .crate_type("lib")
        .input("foo.rs")
        .arg(format!("-Zdump-mono-stats={}", cwd().display()))
        .arg("-Zdump-mono-stats-format=json")
        .run();
    assert!(rfs::read_to_string("foo.mono_items.json").contains(r#""name":"bar""#));
}
