// check-pass
// compile-flags:--test
// normalize-stdout-test: "src/test/rustdoc-ui" -> "$$DIR"

#![feature(cfg_doctest)]

// Make sure `cfg(doctest)` is set when finding doctests but not inside
// the doctests.

/// ```
/// #![feature(cfg_doctest)]
/// assert!(!cfg!(doctest));
/// ```
#[cfg(doctest)]
pub struct Foo;
