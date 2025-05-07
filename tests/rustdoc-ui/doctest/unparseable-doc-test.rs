//@ compile-flags: --test
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ failure-status: 101
//@ rustc-env: RUST_BACKTRACE=0

/// ```rust
/// let x = 7;
/// "unterminated
/// ```
pub fn foo() {}
