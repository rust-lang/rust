// ignore-tidy-linelength
#![deny(broken_intra_doc_links)]

pub mod char {}

/// See also [type@char]
// @has intra_link_prim_precedence/struct.MyString.html '//a/@href' 'https://doc.rust-lang.org/nightly/std/primitive.char.html'
pub struct MyString;

/// See also [char]
// @has intra_link_prim_precedence/struct.MyString2.html '//a/@href' 'intra_link_prim_precedence/char/index.html'
pub struct MyString2;
