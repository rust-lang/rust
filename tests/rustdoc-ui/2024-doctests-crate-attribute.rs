//@ check-pass
//@ edition: 2024
//@ compile-flags: --test --test-args=--test-threads=1
//@ normalize-stdout-test: "tests/rustdoc-ui" -> "$$DIR"
//@ normalize-stdout-test: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout-test: ".rs:\d+:\d+" -> ".rs:$$LINE:$$COL"

/// This doctest is used to ensure that if a crate attribute is present,
/// it will not be part of the merged doctests.
///
/// ```
/// #![doc(html_playground_url = "foo")]
///
/// pub struct Bar;
/// ```
///
/// This one will allow us to confirm that the doctest above will be a
/// standalone one (there will be two separate doctests passes).
///
/// ```
/// let x = 12;
/// ```
pub struct Foo;
