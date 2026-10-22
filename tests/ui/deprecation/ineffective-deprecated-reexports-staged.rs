//! The lint fires in `staged_api` crates too; the standard library carries two such re-exports.
//@ check-pass

#![feature(staged_api)]
#![stable(feature = "test", since = "1.0.0")]
#![allow(unused_imports)]

#[stable(feature = "test", since = "1.0.0")]
pub struct Item;

#[stable(feature = "test", since = "1.0.0")]
#[deprecated(since = "1.1.0", note = "moved")] //~ WARN `#[deprecated]` on a `use` item has no effect
pub use self::Item as Moved;

#[stable(feature = "test", since = "1.0.0")]
#[deprecated(since = "TBD", note = "will move")] //~ WARN `#[deprecated]` on a `use` item has no effect
pub use self::Item as WillMove;

fn main() {}
