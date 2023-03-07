#![crate_type = "lib"]
#![crate_name = "summaries"]

#![allow(rustdoc::broken_intra_doc_links)]

//! This *summary* has a [link], [`code`], and [`Sidebar2`] intra-doc.
//!
//! This is the second paragraph. It should not be rendered.
//! To test that intra-doc links are resolved properly, [`code`] should render
//! the square brackets, and [`Sidebar2`] should not.
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
