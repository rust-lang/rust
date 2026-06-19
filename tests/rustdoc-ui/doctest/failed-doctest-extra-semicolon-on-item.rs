// FIXME: if/when the output of the test harness can be tested on its own, this test should be
// adapted to use that, and that normalize line can go away
// Regression test for #157371. The warning for a trailing semicolon after an item should
// point inside the doctest, not at unrelated source following the documentation.

//@ compile-flags:--test
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

/// <https://github.com/rust-lang/rust/issues/91014>
/// <https://github.com/rust-lang/rust/issues/157371>
///
/// ```rust
/// struct S {};
//~^ WARN the `main` function of this doctest won't be run
///
/// fn main() {
///    assert_eq!(0, 1);
/// }
/// ```
mod m {
    // This line should not be highlighted by the doctest warning.
}
