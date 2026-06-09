//@ compile-flags: --test
//@ check-test-line-numbers-match

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
