// ignore-tidy-linelength
#![deny(intra_doc_link_resolution_failure)]

pub mod char {}

/// See also [type@char]
// @has intra_link_prim_precedence/struct.MyString.html '//a/@href' 'https://doc.rust-lang.org/nightly/std/primitive.char.html'
pub struct MyString;

/// See also [char]
// @has intra_link_prim_precedence/struct.MyString2.html '//a/@href' 'intra_link_prim_precedence/char/index.html'
pub struct MyString2;
