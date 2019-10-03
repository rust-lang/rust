// build-pass
// compile-flags:--test
// normalize-stdout-test: "src/test/rustdoc-ui" -> "$$DIR"

#![feature(doc_cfg)]

// Make sure `cfg(rustdoc)` is set when finding doctests but not inside the doctests.

/// ```
/// #![feature(doc_cfg)]
/// assert!(!cfg!(rustdoc));
/// ```
#[cfg(rustdoc)]
pub struct Foo;
