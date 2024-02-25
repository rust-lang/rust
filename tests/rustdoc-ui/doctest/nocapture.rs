//@ check-pass
//@ compile-flags:--test -Zunstable-options --nocapture
//@ normalize-stdout-test: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"

/// ```
/// println!("hello!");
/// eprintln!("stderr");
/// ```
pub struct Foo;
