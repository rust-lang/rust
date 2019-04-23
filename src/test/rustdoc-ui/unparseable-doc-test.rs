// compile-flags: --test
// normalize-stdout-test: "src/test/rustdoc-ui" -> "$$DIR"
// failure-status: 101
// rustc-env: RUST_BACKTRACE=0

/// ```rust
/// let x = 7;
/// "unterminated
/// ```
pub fn foo() {}
