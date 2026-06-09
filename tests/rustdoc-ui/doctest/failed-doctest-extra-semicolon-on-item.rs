// FIXME: if/when the output of the test harness can be tested on its own, this test should be
// adapted to use that, and that normalize line can go away

//@ compile-flags:--test
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

/// <https://github.com/rust-lang/rust/issues/91014>
///
/// ```rust
//~^ WARN the `main` function of this doctest won't be run
/// struct S {};
///
/// fn main() {
///    assert_eq!(0, 1);
/// }
/// ```
mod m {}
