//@ compile-flags:--test --test-args=--test-threads=1
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

/// ```
/// #![deprecated(since = "5.2", note = "foo was rarely used. \
///    Users should instead use bar")]
/// ```
pub fn f() {}
