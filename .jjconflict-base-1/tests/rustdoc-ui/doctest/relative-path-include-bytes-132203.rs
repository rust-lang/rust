//@ revisions: edition2015 edition2024
//@[edition2015]edition:2015
//@[edition2015]check-fail
//@[edition2015]failure-status: 101
//@[edition2015]compile-flags:--test --test-args=--test-threads=1
//@[edition2024]edition:2024
//@[edition2024]check-pass
//@[edition2024]compile-flags:--test --test-args=--test-threads=1
//@ normalize-stdout: "tests.rustdoc-ui.doctest." -> "$$DIR/"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout: "`: .* \(os error 2\)" -> "`: $$FILE_NOT_FOUND_MSG (os error 2)"
//@ normalize-stdout: "ran in \d+\.\d+s" -> "ran in $$TIME"
//@ normalize-stdout: "compilation took \d+\.\d+s" -> "compilation took $$TIME"

// https://github.com/rust-lang/rust/issues/132203
// This version, because it's edition2024, passes thanks to the new
// relative path. The edition2015 version fails, because paths are
// resolved relative to the rs file instead of relative to the md file.

#![doc=include_str!("auxiliary/relative-dir.md")]
