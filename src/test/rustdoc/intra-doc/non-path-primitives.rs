// ignore-tidy-linelength
#![crate_name = "foo"]
#![feature(intra_doc_pointers)]
#![deny(broken_intra_doc_links)]

// @has foo/index.html '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.slice.html#method.rotate_left"]' 'slice::rotate_left'
//! [slice::rotate_left]

// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.array.html#method.map"]' 'array::map'
//! [array::map]

// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.str.html"]' 'owned str'
// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.str.html"]' 'str ref'
// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.str.html#method.is_empty"]' 'str::is_empty'
// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.str.html#method.len"]' '&str::len'
//! [owned str][str]
//! [str ref][&str]
//! [str::is_empty]
//! [&str::len]

// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.pointer.html#method.is_null"]' 'pointer::is_null'
// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.pointer.html#method.is_null"]' '*const::is_null'
// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.pointer.html#method.is_null"]' '*mut::is_null'
//! [pointer::is_null]
//! [*const::is_null]
//! [*mut::is_null]

// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.unit.html"]' 'unit'
//! [unit]

// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.tuple.html"]' 'tuple'
//! [tuple]

// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.reference.html"]' 'reference'
// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.reference.html"]' '&'
// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.reference.html"]' '&mut'
//! [reference]
//! [&]
//! [&mut]

// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.fn.html"]' 'fn'
//! [fn]

// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.never.html"]' 'never'
// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.never.html"]' '!'
//! [never]
//! [!]
