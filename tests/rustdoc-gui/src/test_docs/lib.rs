//! The point of this crate is to be able to have enough different "kinds" of
//! documentation generated so we can test each different features.
#![doc(html_playground_url="https://play.rust-lang.org/")]

#![crate_name = "test_docs"]
#![feature(rustdoc_internals)]
#![feature(doc_cfg)]
#![feature(associated_type_defaults)]

/*!
Enable the feature <span class="stab portability"><code>some-feature</code></span> to enjoy
this crate even more!
Enable the feature <span class="stab portability"><code>some-feature</code></span> to enjoy
this crate even more!
Enable the feature <span class="stab portability"><code>some-feature</code></span> to enjoy
this crate even more!

Also, stop using `bar` as it's <span class="stab deprecated" title="">deprecated</span>.
Also, stop using `bar` as it's <span class="stab deprecated" title="">deprecated</span>.
Also, stop using `bar` as it's <span class="stab deprecated" title="">deprecated</span>.

Finally, you can use `quz` only on <span class="stab portability"><code>Unix or x86-64</code>
</span>.
Finally, you can use `quz` only on <span class="stab portability"><code>Unix or x86-64</code>
</span>.
*/

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
/// A failing to run one:
///
/// ```should_panic
/// panic!("tadam");
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
#[non_exhaustive]
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

/// Very long code text [`hereIgoWithLongTextBecauseWhyNotAndWhyWouldntI`][lnk].
///
/// [lnk]: crate::long_code_block_link
pub mod long_code_block_link {}

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
///
/// #### You know the drill.
///
/// More text.
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

pub trait EmptyTrait1 {}
pub trait EmptyTrait2 {}
pub trait EmptyTrait3 {}

pub struct HasEmptyTraits{}

impl EmptyTrait1 for HasEmptyTraits {}
impl EmptyTrait2 for HasEmptyTraits {}
#[doc(cfg(feature = "some-feature"))]
impl EmptyTrait3 for HasEmptyTraits {}

mod macros;
pub use macros::*;

#[doc(alias = "AliasForTheStdReexport")]
pub use ::std as TheStdReexport;

pub mod details {
    /// We check the appearance of the `<details>`/`<summary>` in here.
    ///
    /// ## Hello
    ///
    /// <details>
    /// <summary><h4>I'm a summary</h4></summary>
    /// <div>I'm the content of the details!</div>
    /// </details>
    pub struct Details;

    impl Details {
        /// We check the appearance of the `<details>`/`<summary>` in here.
        ///
        /// ## Hello
        ///
        /// <details>
        /// <summary><h4>I'm a summary</h4></summary>
        /// <div>I'm the content of the details!</div>
        /// </details>
        pub fn method() {}
    }
}

pub mod doc_block_table {

    pub trait DocBlockTableTrait {
        fn foo();
    }

    /// Struct doc.
    ///
    /// | header1                  | header2                  |
    /// |--------------------------|--------------------------|
    /// | Lorem Ipsum, Lorem Ipsum | Lorem Ipsum, Lorem Ipsum |
    /// | Lorem Ipsum, Lorem Ipsum | Lorem Ipsum, Lorem Ipsum |
    /// | Lorem Ipsum, Lorem Ipsum | Lorem Ipsum, Lorem Ipsum |
    /// | Lorem Ipsum, Lorem Ipsum | Lorem Ipsum, Lorem Ipsum |
    pub struct DocBlockTable {}

    impl DocBlockTableTrait for DocBlockTable {
        /// Trait impl func doc for struct.
        ///
        /// | header1                  | header2                  |
        /// |--------------------------|--------------------------|
        /// | Lorem Ipsum, Lorem Ipsum | Lorem Ipsum, Lorem Ipsum |
        fn foo() {
            println!();
        }
    }

}

pub struct NotableStructWithLongName<R>(R);

impl<R: std::io::Read> NotableStructWithLongName<R> {
    pub fn create_an_iterator_from_read(r: R) -> NotableStructWithLongName<R> { Self(r) }
}

impl<R: std::io::Read> std::iter::Iterator for NotableStructWithLongName<R> {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> { () }
}

pub trait TraitWithNoDocblocks {
    fn first_fn(&self);
    fn second_fn(&self);
}

pub struct TypeWithNoDocblocks;

impl TypeWithNoDocblocks {
    fn x() -> Option<Self> {
        Some(Self)
    }
    fn y() -> Option<u32> {
        // code comment
        let t = Self::x()?;
        Some(0)
    }
}

impl TypeWithNoDocblocks {
    pub fn first_fn(&self) {}
    pub fn second_fn<'a>(&'a self) {
        let x = 12;
        let y = "a";
        let z = false;
    }
}

pub unsafe fn unsafe_fn() {}

pub fn safe_fn() {}

#[repr(C)]
pub struct WithGenerics<T: TraitWithNoDocblocks, S = String, E = WhoLetTheDogOut, P = i8> {
    s: S,
    t: T,
    e: E,
    p: P,
}

pub struct StructWithPublicUndocumentedFields {
    pub first: u32,
    pub second: u32,
}

pub const CONST: u8 = 0;

pub trait TraitWithoutGenerics {
    const C: u8 = CONST;
    type T = SomeType;

    fn foo();
}

pub mod trait_members {
    pub trait TraitMembers {
        /// Some type
        type Type;
        /// Some function
        fn function();
        /// Some other function
        fn function2();
    }
    pub struct HasTrait;
    impl TraitMembers for HasTrait {
        type Type = u8;
        fn function() {}
        fn function2() {}
    }
}

pub struct TypeWithImplDoc;

/// impl doc
impl TypeWithImplDoc {
    /// fn doc
    pub fn test_fn() {}
}

/// <sub id="codeblock-sub-1">
///
/// ```
/// one
/// ```
///
/// </sub>
///
/// <sub id="codeblock-sub-3">
///
/// ```
/// one
/// two
/// three
/// ```
///
/// </sub>
pub mod codeblock_sub {}
pub mod search_results {

    pub struct SearchResults {
        pub foo: i32,
    }

    #[macro_export]
    macro_rules! foo {
        () => {};
    }

}
