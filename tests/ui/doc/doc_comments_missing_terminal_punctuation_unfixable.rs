#![feature(custom_inner_attributes)]
#![rustfmt::skip]
#![warn(clippy::doc_comments_missing_terminal_punctuation)]
// Only line doc comments are provided with suggestions.
//@no-rustfix

/// Sometimes a doc attribute is used for concatenation
/// ```
#[doc = ""]
/// ```
//~^ doc_comments_missing_terminal_punctuation
struct DocAttribute;

