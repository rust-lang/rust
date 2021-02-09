//! This module contains some useful constants.

#![deny(clippy::missing_docs_in_private_items)]

/// List of the built-in types names.
///
/// See also [the reference][reference-types] for a list of such types.
///
/// [reference-types]: https://doc.rust-lang.org/reference/types.html
pub const BUILTIN_TYPES: &[&str] = &[
    "i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "i128", "u128", "isize", "usize", "f32", "f64", "bool",
    "str", "char",
];
