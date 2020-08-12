// ignore-tidy-linelength
#![deny(broken_intra_doc_links)]

pub mod char {}

/// See also [type@char]
// @has intra_link_prim_precedence/struct.MyString.html '//a/@href' 'https://doc.rust-lang.org/nightly/std/primitive.char.html'
pub struct MyString;

/// See also [char]
// @has intra_link_prim_precedence/struct.MyString2.html '//a/@href' 'https://doc.rust-lang.org/nightly/std/primitive.char.html'
pub struct MyString2;

/// See also [crate::char] and [mod@char]
// @has intra_link_prim_precedence/struct.MyString3.html '//*[@href="../intra_link_prim_precedence/char/index.html"]' 'crate::char'
// @has - '//*[@href="../intra_link_prim_precedence/char/index.html"]' 'mod@char'
pub struct MyString3;
