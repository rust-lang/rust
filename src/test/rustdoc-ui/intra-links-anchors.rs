#![deny(broken_intra_doc_links)]

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
//~^ ERROR `Foo::f#hola` contains an anchor
pub fn foo() {}

/// Empty.
///
/// Another anchor error: [hello#people#!].
//~^ ERROR `hello#people#!` contains multiple anchors
pub fn bar() {}

/// Empty?
///
/// Damn enum's variants: [Enum::A#whatever].
//~^ ERROR `Enum::A#whatever` contains an anchor
pub fn enum_link() {}

/// Primitives?
///
/// [u32#hello]
//~^ ERROR `u32#hello` contains an anchor
pub fn x() {}
