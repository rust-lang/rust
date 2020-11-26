// check-pass
// compile-flags:--test
// normalize-stdout-test: "src/test/rustdoc-ui" -> "$$DIR"

// Make sure `cfg(doctest)` is set when finding doctests but not inside
// the doctests.

/// ```
/// assert!(!cfg!(doctest));
/// ```
#[cfg(doctest)]
pub struct Foo;
