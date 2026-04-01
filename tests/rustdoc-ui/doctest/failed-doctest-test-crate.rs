// FIXME: if/when the output of the test harness can be tested on its own, this test should be
// adapted to use that, and that normalize line can go away

//@ revisions: edition2015 edition2024
//@[edition2015]edition:2015
//@[edition2024]edition:2024
//@ compile-flags:--test
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout: "ran in \d+\.\d+s" -> "ran in $$TIME"
//@ normalize-stdout: "compilation took \d+\.\d+s" -> "compilation took $$TIME"
//@ failure-status: 101

/// <https://github.com/rust-lang/rust/pull/137899#discussion_r1976743383>
///
/// ```rust
/// use test::*;
/// ```
pub mod m {}
