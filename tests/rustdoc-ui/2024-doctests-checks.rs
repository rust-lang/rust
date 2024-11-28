//@ check-pass
//@ edition: 2024
//@ compile-flags: --test --test-args=--test-threads=1
//@ normalize-stdout-test: "tests/rustdoc-ui" -> "$$DIR"
//@ normalize-stdout-test: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout-test: ".rs:\d+:\d+" -> ".rs:$$LINE:$$COL"

/// ```
/// let x = 12;
/// ```
///
/// This one should not be a merged doctest (because of `$crate`). The output
/// will confirm it by displaying both merged and standalone doctest passes.
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
