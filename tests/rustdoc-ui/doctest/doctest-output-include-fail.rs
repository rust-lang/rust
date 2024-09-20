//@ compile-flags:--test --test-args=--test-threads=1
//@ normalize-stdout-test: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout-test: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ failure-status: 101

// https://github.com/rust-lang/rust/issues/130470
#![doc = include_str!("doctest-output-include-fail.md")]
