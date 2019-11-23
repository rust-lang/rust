#![deny(intra_doc_link_resolution_failure)]

// A few tests on anchors.

/// Hello people.
///
/// You can anchors? Here's one!
///
/// # hola
///
/// Isn't it amazing?
pub struct Foo {
    pub f: u8,
}

pub enum Enum {
    A,
    B,
}

/// Have you heard about stuff?
///
/// Like [Foo#hola].
///
/// Or maybe [Foo::f#hola].
//~^ ERROR `[Foo::f#hola]` has an anchor issue...
pub fn foo() {}

/// Empty.
///
/// Another anchor error: [hello#people#!].
//~^ ERROR `[hello#people#!]` has an anchor issue...
pub fn bar() {}

/// Empty?
///
/// Damn enum's variants: [Enum::A#whatever].
//~^ ERROR `[Enum::A#whatever]` has an anchor issue...
pub fn enum_link() {}

/// Primitives?
///
/// [u32#hello]
//~^ ERROR `[u32#hello]` has an anchor issue...
pub fn x() {}
