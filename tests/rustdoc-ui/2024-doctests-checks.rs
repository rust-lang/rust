//@ check-pass
//@ compile-flags: --test --test-args=--test-threads=1 -Zunstable-options --edition 2024
//@ normalize-stdout-test: "tests/rustdoc-ui" -> "$$DIR"
//@ normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout-test ".rs:\d+:\d+" -> ".rs:$$LINE:$$COL"

/// This one should fail: crate attributes should remain crate attributes
/// in standalone doctests.
///
/// ```compile_fail
/// #![deny(missing_docs)]
///
/// pub struct Bar;
/// ```
///
/// This one should not impact the other merged doctests.
///
/// ```
/// #![deny(unused)]
/// ```
///
/// ```
/// let x = 12;
/// ```
///
/// This one should not be a merged doctest (because of `$crate`):
///
/// ```
/// macro_rules! bla {
///     () => {{
///         $crate::foo();
///     }}
/// }
///
/// fn foo() {}
///
/// fn main() {
///     bla!();
/// }
/// ```
pub struct Foo;
