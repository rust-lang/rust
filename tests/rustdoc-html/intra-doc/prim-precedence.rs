#![deny(rustdoc::broken_intra_doc_links)]

pub mod char {
    /// [char]
    //@ has prim_precedence/char/type.Inner.html '//a/@href' '{{channel}}/std/primitive.char.html'
    pub struct Inner;
}

/// See [prim@char]
//@ has prim_precedence/type.MyString.html '//a/@href' '{{channel}}/std/primitive.char.html'
pub struct MyString;

/// See also [crate::char] and [mod@char]
//@ has prim_precedence/type.MyString2.html '//*[@href="char/index.html"]' 'crate::char'
//@ has - '//*[@href="char/index.html"]' 'char'
pub struct MyString2;
