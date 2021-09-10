//! The point of this crate is to be able to have enough different "kinds" of
//! documentation generated so we can test each different features.

#![crate_name = "test_docs"]
#![feature(doc_keyword)]
#![feature(doc_cfg)]

use std::convert::AsRef;
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
///
/// An inlined `code`!
pub fn foo() {}

/// Just a normal struct.
pub struct Foo;

impl Foo {
    #[must_use]
    pub fn must_use(&self) -> bool {
        true
    }
}

impl AsRef<str> for Foo {
    fn as_ref(&self) -> &str {
        "hello"
    }
}

/// Just a normal enum.
#[doc(alias = "ThisIsAnAlias")]
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
    /// Some func 3.
    fn func3();

    /// Some func 1.
    fn func1();

    fn another();
    fn why_not();

    /// Some func 2.
    fn func2();

    fn hello();
}

/// ```compile_fail
/// whatever
/// ```
///
/// Check for "i" signs in lists!
///
/// 1. elem 1
/// 2. test 1
///    ```compile_fail
///    fn foo() {}
///    ```
/// 3. elem 3
/// 4. ```ignore (it's a test)
///    fn foo() {}
///    ```
/// 5. elem 5
///
/// Final one:
///
/// ```ignore (still a test)
/// let x = 12;
/// ```
pub fn check_list_code_block() {}

/// a thing with a label
#[deprecated(since = "1.0.0", note = "text why this deprecated")]
#[doc(cfg(unix))]
pub fn replaced_function() {}

/// Some doc with `code`!
pub enum AnEnum {
    WithVariants { and: usize, sub: usize, variants: usize },
}

#[doc(keyword = "CookieMonster")]
/// Some keyword.
pub mod keyword {}

/// Just some type alias.
pub type SomeType = u32;

pub mod huge_amount_of_consts {
    include!(concat!(env!("OUT_DIR"), "/huge_amount_of_consts.rs"));
}
