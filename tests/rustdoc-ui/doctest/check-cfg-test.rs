// check-pass
// compile-flags: --test --nocapture --check-cfg=cfg(feature,values("test")) -Z unstable-options
// normalize-stderr-test: "tests/rustdoc-ui/doctest" -> "$$DIR"
// normalize-stdout-test: "tests/rustdoc-ui/doctest" -> "$$DIR"
// normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"

/// The doctest will produce a warning because feature invalid is unexpected
/// ```
/// #[cfg(feature = "invalid")]
/// assert!(false);
/// ```
pub struct Foo;
