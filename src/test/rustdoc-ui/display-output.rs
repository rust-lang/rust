// check-pass
// compile-flags:-Zunstable-options --display-doctest-warnings --test
// normalize-stdout-test: "src/test/rustdoc-ui" -> "$$DIR"
// normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"

/// ```
/// let x = 12;
/// ```
pub fn foo() {}
