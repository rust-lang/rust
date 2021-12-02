//! The point of this crate is to be able to have enough different "kinds" of
//! documentation generated so we can test each different features.

#![crate_name = "test_docs"]
#![feature(rustdoc_internals)]
#![feature(doc_cfg)]

use std::convert::AsRef;
use std::fmt;

/// Basic function with some code examples:
///
/// ```
/// println!("nothing fancy");
/// println!("but with two lines!");
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
///
/// # title!
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

/// Very long code text `hereIgoWithLongTextBecauseWhyNotAndWhyWouldntI`.
pub mod long_code_block {}

#[macro_export]
macro_rules! repro {
    () => {};
}

pub use crate::repro as repro2;

/// # Top-doc Prose title
///
/// Text below title.
///
/// ## Top-doc Prose sub-heading
///
/// Text below sub-heading.
///
/// ### Top-doc Prose sub-sub-heading
///
/// Text below sub-sub-heading
pub struct HeavilyDocumentedStruct {
    /// # Title for field
    /// ## Sub-heading for field
    pub nothing: (),
}

/// # Title for struct impl doc
///
/// Text below heading.
///
/// ## Sub-heading for struct impl doc
///
/// Text below sub-heading.
///
/// ### Sub-sub-heading for struct impl doc
///
/// Text below sub-sub-heading.
///
impl HeavilyDocumentedStruct {
    /// # Title for struct impl-item doc
    /// Text below title.
    /// ## Sub-heading for struct impl-item doc
    /// Text below sub-heading.
    /// ### Sub-sub-heading for struct impl-item doc
    /// Text below sub-sub-heading.
    pub fn do_nothing() {}
}

/// # Top-doc Prose title
///
/// Text below title.
///
/// ## Top-doc Prose sub-heading
///
/// Text below sub-heading.
///
/// ### Top-doc Prose sub-sub-heading
///
/// Text below sub-sub-heading
pub enum HeavilyDocumentedEnum {
    /// # None prose title
    /// ## None prose sub-heading
    None,
    /// # Wrapped prose title
    /// ## Wrapped prose sub-heading
    Wrapped(
        /// # Wrapped.0 prose title
        /// ## Wrapped.0 prose sub-heading
        String,
        String,
    ),
    Structy {
        /// # Structy prose title
        /// ## Structy prose sub-heading
        alpha: String,
        beta: String,
    },
}

/// # Title for enum impl doc
///
/// Text below heading.
///
/// ## Sub-heading for enum impl doc
///
/// Text below sub-heading.
///
/// ### Sub-sub-heading for enum impl doc
///
/// Text below sub-sub-heading.
///
impl HeavilyDocumentedEnum {
    /// # Title for enum impl-item doc
    /// Text below title.
    /// ## Sub-heading for enum impl-item doc
    /// Text below sub-heading.
    /// ### Sub-sub-heading for enum impl-item doc
    /// Text below sub-sub-heading.
    pub fn do_nothing() {}
}

/// # Top-doc prose title
///
/// Text below heading.
///
/// ## Top-doc prose sub-heading
///
/// Text below heading.
pub union HeavilyDocumentedUnion {
    /// # Title for union variant
    /// ## Sub-heading for union variant
    pub nothing: (),
    pub something: f32,
}

/// # Title for union impl doc
/// ## Sub-heading for union impl doc
impl HeavilyDocumentedUnion {
    /// # Title for union impl-item doc
    /// ## Sub-heading for union impl-item doc
    pub fn do_nothing() {}
}

/// # Top-doc prose title
///
/// Text below heading.
///
/// ## Top-doc prose sub-heading
///
/// Text below heading.
#[macro_export]
macro_rules! heavily_documented_macro {
    () => {};
}
