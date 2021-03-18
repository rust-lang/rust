//! The point of this crate is to be able to have enough different "kinds" of
//! documentation generated so we can test each different features.

#![crate_name = "test_docs"]

use std::fmt;

/// Basic function with some code examples:
///
/// ```
/// println!("nothing fancy");
/// ```
///
/// A failing to compile one:
///
/// ```compile_fail
/// println!("where did my argument {} go? :'(");
/// ```
///
/// An ignored one:
///
/// ```ignore (it's a test)
/// Let's say I'm just some text will ya?
/// ```
pub fn foo() {}

/// Just a normal struct.
pub struct Foo;

impl Foo {
    #[must_use]
    pub fn must_use(&self) -> bool { true }
}

/// Just a normal enum.
pub enum WhoLetTheDogOut {
    /// Woof!
    Woof,
    /// Meoooooooow...
    Meow,
}

/// Who doesn't love to wrap a `format!` call?
pub fn some_more_function<T: fmt::Debug>(t: &T) -> String {
    format!("{:?}", t)
}

/// Woohoo! A trait!
pub trait AnotherOne {
    /// Some func 1.
    fn func1();

    /// Some func 2.
    fn func2();

    /// Some func 3.
    fn func3();
}

/// Check for "i" signs in lists!
///
/// 1. elem 1
/// 2.test 1
///   ```compile_fail
///   fn foo() {}
///   ```
/// 3. elem 3
/// 4. ```ignore (it's a test)
///    fn foo() {}
///    ```
/// 5. elem 5
pub fn check_list_code_block() {}
