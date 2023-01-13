#![crate_type = "lib"]
#![crate_name = "summaries"]

//! This *summary* has a [link] and `code`.
//!
//! This is the second paragraph.
//!
//! [link]: https://example.com

// @hasraw search-index.js 'This <em>summary</em> has a link and <code>code</code>.'
// @!hasraw - 'second paragraph'

/// This `code` will be rendered in a code tag.
///
/// This text should not be rendered.
pub struct Sidebar;

// @hasraw search-index.js 'This <code>code</code> will be rendered in a code tag.'
// @hasraw summaries/sidebar-items.js 'This `code` will be rendered in a code tag.'
// @!hasraw - 'text should not be rendered'

/// ```text
/// this block should not be rendered
/// ```
pub struct Sidebar2;

// @!hasraw summaries/sidebar-items.js 'block should not be rendered'
