#![crate_name = "foo"]
#![feature(intra_doc_pointers)]
#![deny(rustdoc::broken_intra_doc_links)]

//@ has foo/index.html '//a[@href="{{channel}}/std/primitive.slice.html#method.rotate_left"]' 'slice::rotate_left'
//! [slice::rotate_left]

//@ has - '//a[@href="{{channel}}/std/primitive.array.html#method.map"]' 'array::map'
//! [array::map]

//@ has - '//a[@href="{{channel}}/std/primitive.str.html"]' 'owned str'
//@ has - '//a[@href="{{channel}}/std/primitive.str.html"]' 'str ref'
//@ has - '//a[@href="{{channel}}/std/primitive.str.html#method.is_empty"]' 'str::is_empty'
//@ has - '//a[@href="{{channel}}/std/primitive.str.html#method.len"]' '&str::len'
//! [owned str][str]
//! [str ref][&str]
//! [str::is_empty]
//! [&str::len]

//@ has - '//a[@href="{{channel}}/std/primitive.pointer.html#method.is_null"]' 'pointer::is_null'
//@ has - '//a[@href="{{channel}}/std/primitive.pointer.html#method.is_null"]' '*const::is_null'
//@ has - '//a[@href="{{channel}}/std/primitive.pointer.html#method.is_null"]' '*mut::is_null'
//! [pointer::is_null]
//! [*const::is_null]
//! [*mut::is_null]

//@ has - '//a[@href="{{channel}}/std/primitive.unit.html"]' 'unit'
//! [unit]

//@ has - '//a[@href="{{channel}}/std/primitive.tuple.html"]' 'tuple'
//! [tuple]

//@ has - '//a[@href="{{channel}}/std/primitive.reference.html"]' 'reference'
//@ has - '//a[@href="{{channel}}/std/primitive.reference.html"]' '&'
//@ has - '//a[@href="{{channel}}/std/primitive.reference.html"]' '&mut'
//! [reference]
//! [&]
//! [&mut]

//@ has - '//a[@href="{{channel}}/std/primitive.fn.html"]' 'fn'
//! [fn]

//@ has - '//a[@href="{{channel}}/std/primitive.never.html"]' 'never'
//@ has - '//a[@href="{{channel}}/std/primitive.never.html"]' '!'
//! [never]
//! [!]
