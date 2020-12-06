// check-pass
// compile-flags:--test
// normalize-stdout-test: "src/test/rustdoc-ui" -> "$$DIR"
// normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"

// The literal `mod` in the doctest comments makes rustdoc emit a bogus
// `extern crate mod;` line which would break the doctests.
#![doc(test(no_crate_inject))]

/// ```
/// // ^ mod.rs line 10
/// assert_eq!(1, 1);
/// ```
pub mod a;
