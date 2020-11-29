// ignore-tidy-linelength
#![deny(broken_intra_doc_links)]

pub mod char {
    /// [char]
    // @has prim_precedence/char/struct.Inner.html '//a/@href' 'https://doc.rust-lang.org/nightly/std/primitive.char.html'
    pub struct Inner;
}

/// See [prim@char]
// @has prim_precedence/struct.MyString.html '//a/@href' 'https://doc.rust-lang.org/nightly/std/primitive.char.html'
pub struct MyString;

/// See also [crate::char] and [mod@char]
// @has prim_precedence/struct.MyString2.html '//*[@href="../prim_precedence/char/index.html"]' 'crate::char'
// @has - '//*[@href="../prim_precedence/char/index.html"]' 'mod@char'
pub struct MyString2;
