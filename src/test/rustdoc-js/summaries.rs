#![crate_type = "lib"]
#![crate_name = "summaries"]

//! This *summary* has a [link] and `code`.
//!
//! This is the second paragraph.
//!
//! [link]: https://example.com

/// This `code` will be rendered in a code tag.
///
/// This text should not be rendered.
pub struct Sidebar;

/// ```text
/// this block should not be rendered
/// ```
pub struct Sidebar2;
