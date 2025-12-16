//@ check-pass
//@ compile-flags:--test
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"

// Make sure `cfg(doc)` is set when finding doctests but not inside the doctests.

/// ```
/// assert!(!cfg!(doc));
/// ```
#[cfg(doc)]
pub struct Foo;
