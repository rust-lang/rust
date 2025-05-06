// We need this option to be enabled for the `foo` macro declaration to ensure
// that the link on the ident is not including whitespace characters.

//@ compile-flags: -Zunstable-options --generate-link-to-definition
#![crate_name = "foo"]

//@ has 'src/foo/source-code-highlight.rs.html'

//@ hasraw - '<a href="../../foo/macro.foo.html">foo</a>'
#[macro_export]
macro_rules! foo {
    () => {}
}

//@ hasraw - '<span class="macro">foo!</span>'
foo! {}

//@ hasraw - '<a href="../../foo/fn.f.html">f</a>'
#[rustfmt::skip]
pub fn f () {}
//@ hasraw - '<a href="../../foo/struct.Bar.html">Bar</a>'
//@ hasraw - '<a href="../../foo/struct.Bar.html">Bar</a>'
//@ hasraw - '<a href="{{channel}}/std/primitive.u32.html">u32</a>'
#[rustfmt::skip]
pub struct Bar ( u32 );
//@ hasraw - '<a href="../../foo/enum.Foo.html">Foo</a>'
pub enum Foo {
    A,
}
