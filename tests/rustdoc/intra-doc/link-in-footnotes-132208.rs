// Rustdoc has multiple passes and if the footnote pass is run before the link replacer
// one, intra doc links are not generated inside footnote definitions. This test
// therefore ensures that intra-doc link are correctly generated inside footnote
// definitions.
//
// Regression test for <https://github.com/rust-lang/rust/issues/132208>.

#![crate_name = "foo"]

//@ has 'foo/index.html'
//@ has - '//*[@class="docblock"]//a[@href="struct.Bar.html"]' 'a'
//@ has - '//*[@class="docblock"]//*[@class="footnotes"]//a[@href="struct.Foo.html"]' 'b'

//! [a]: crate::Bar
//! [b]: crate::Foo
//!
//! link in body: [a]
//!
//! see footnote[^1]
//!
//! [^1]: link in footnote: [b]

pub struct Bar;
pub struct Foo;
