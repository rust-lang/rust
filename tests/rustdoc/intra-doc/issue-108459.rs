#![deny(rustdoc::broken_intra_doc_links)]

pub struct S;
pub mod char {}

// Ensure this doesn't ICE due to trying to slice off non-existent backticks from "S"

/// See [S] and [`S`]
pub struct MyStruct1;

// Ensure that link texts are replaced correctly even if there are multiple links with
// the same target but different text

/// See also [crate::char] and [mod@char] and [prim@char]
// @has issue_108459/struct.MyStruct2.html '//*[@href="char/index.html"]' 'crate::char'
// @has - '//*[@href="char/index.html"]' 'char'
// @has - '//*[@href="{{channel}}/std/primitive.char.html"]' 'char'
pub struct MyStruct2;

/// See also [mod@char] and [prim@char] and [crate::char]
// @has issue_108459/struct.MyStruct3.html '//*[@href="char/index.html"]' 'crate::char'
// @has - '//*[@href="char/index.html"]' 'char'
// @has - '//*[@href="{{channel}}/std/primitive.char.html"]' 'char'
pub struct MyStruct3;

// Ensure that links are correct even if there are multiple links with the same text but
// different targets

/// See also [char][mod@char] and [char][prim@char]
// @has issue_108459/struct.MyStruct4.html '//*[@href="char/index.html"]' 'char'
// @has - '//*[@href="{{channel}}/std/primitive.char.html"]' 'char'
pub struct MyStruct4;

/// See also [char][prim@char] and [char][crate::char]
// @has issue_108459/struct.MyStruct5.html '//*[@href="char/index.html"]' 'char'
// @has - '//*[@href="{{channel}}/std/primitive.char.html"]' 'char'
pub struct MyStruct5;
