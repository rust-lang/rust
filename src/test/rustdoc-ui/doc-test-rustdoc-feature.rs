// check-pass
// compile-flags:--test
// normalize-stdout-test: "src/test/rustdoc-ui" -> "$$DIR"
// normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"

#![feature(doc_cfg)]

// Make sure `cfg(doc)` is set when finding doctests but not inside the doctests.

/// ```
/// #![feature(doc_cfg)]
/// assert!(!cfg!(doc));
/// ```
#[cfg(doc)]
pub struct Foo;
