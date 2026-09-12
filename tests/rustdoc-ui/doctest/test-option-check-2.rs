//@ compile-flags: --test --test-args=--test-threads=1
//@ check-test-line-numbers-match
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

#[path = "test-option-check.rs"]
pub mod bar;

/// This is a Foo;
///
/// ```
/// println!("baaaaaar");
/// ```
pub struct Foo;

/// This is a Bar;
///
/// ```
/// println!("fooooo");
/// ```
pub struct Bar;
