// compile-pass
// compile-flags:--test
// normalize-stdout-test: "src/test/rustdoc-ui" -> "$$DIR"

/// this doctest will be ignored:
///
/// ```
/// assert!(false);
/// ```
#[cfg(not(test))]
pub struct Foo;

/// this doctest will be tested:
///
/// ```
/// assert!(true);
/// ```
#[cfg(test)]
pub struct Foo;
