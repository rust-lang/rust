#![crate_type = "lib"]
#![crate_name = "summaries"]

//! This summary has a [link] and `code`.
//!
//! This is the second paragraph.
//!
//! [link]: https://example.com

// @has search-index.js 'This summary has a link and `code`.'
// @!has - 'second paragraph'

/// This `code` should be in backticks.
///
/// This text should not be rendered.
pub struct Sidebar;

// @has summaries/sidebar-items.js 'This `code` should be in backticks.'
// @!has - 'text should not be rendered'

/// ```text
/// this block should not be rendered
/// ```
pub struct Sidebar2;

// @!has summaries/sidebar-items.js 'block should not be rendered'
