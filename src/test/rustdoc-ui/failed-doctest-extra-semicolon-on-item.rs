// FIXME: if/when the output of the test harness can be tested on its own, this test should be
// adapted to use that, and that normalize line can go away

// compile-flags:--test
// normalize-stdout-test: "src/test/rustdoc-ui" -> "$$DIR"
// normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"
// failure-status: 101

/// <https://github.com/rust-lang/rust/issues/91014>
///
/// ```rust
/// struct S {}; // unexpected semicolon after struct def
///
/// fn main() {
///    assert_eq!(0, 1);
/// }
/// ```
mod m {}
