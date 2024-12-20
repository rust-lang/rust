//! Check that we permit a crate-level inner attribute but reject a dangling outer attribute which
//! does not have a following thing that it can target.
//!
//! See <https://doc.rust-lang.org/reference/attributes.html>.

//@ error-pattern: expected item

#![attr = "val"]
#[attr = "val"] // Unterminated
