// compile-flags:-Z unstable-options --show-coverage
// check-pass

#![feature(rustdoc_internals)]

//! the features only used in std also have entries in the table, so make sure those get pulled out
//! properly as well

/// woo, check it out, we can write our own primitive docs lol
#[doc(primitive="unit")]
mod prim_unit {}

/// keywords? sure, pile them on
#[doc(keyword="where")]
mod where_keyword {}
