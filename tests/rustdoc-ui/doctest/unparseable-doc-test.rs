//@ compile-flags: --test
//@ normalize-stdout-test: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ failure-status: 101
//@ rustc-env: RUST_BACKTRACE=0

/// ```rust
/// let x = 7;
/// "unterminated
/// ```
pub fn foo() {}
