// ignore-tidy-linelength
#![crate_name = "foo"]
#![deny(broken_intra_doc_links)]

// @has foo/index.html '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.slice.html#method.rotate_left"]' 'slice::rotate_left'
// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.slice.html#method.rotate_left"]' 'X'
// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.slice.html#method.rotate_left"]' 'Y'
//! [slice::rotate_left]
//! [X]([T]::rotate_left)
//! [Y](&[]::rotate_left)
// These don't work because markdown syntax doesn't allow it.
// [[T]::rotate_left]
//! [&[]::rotate_left]

// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.array.html#method.map"]' 'array::map'
// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.array.html#method.map"]' 'X'
// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.array.html#method.map"]' 'Y'
//! [array::map]
//! [X]([]::map)
//! [Y]([T;N]::map)
// These don't work because markdown syntax doesn't allow it.
// [Z]([T; N]::map)
//! [`[T; N]::map`]
//! [[]::map]
// [Z][]
//
// [Z]: [T; N]::map

// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.pointer.html#method.is_null"]' 'pointer::is_null'
// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.pointer.html#method.is_null"]' '*const::is_null'
// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.pointer.html#method.is_null"]' '*mut::is_null'
// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.pointer.html#method.is_null"]' '*::is_null'
//! [pointer::is_null]
//! [*const::is_null]
//! [*mut::is_null]
//! [*::is_null]

// FIXME: Associated items on some primitives aren't working, because the impls
// are part of the compiler instead of being part of the source code.

// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.unit.html"]' 'unit'
// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.unit.html"]' '()'
//! [unit]
//! [()]

// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.tuple.html"]' 'tuple'
// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.tuple.html"]' 'X'
// @has - '//a[@href="https://doc.rust-lang.org/nightly/std/primitive.tuple.html"]' '(,)'
//! [tuple]
//! [X]((,))
//! [(,)]

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
