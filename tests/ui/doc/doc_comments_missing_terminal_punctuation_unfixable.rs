#![feature(custom_inner_attributes)]
#![rustfmt::skip]
#![warn(clippy::doc_comments_missing_terminal_punctuation)]
// Only line doc comments are provided with suggestions.
//@no-rustfix

/**
//~^ doc_comments_missing_terminal_punctuation
 * Block doc comments work
 *
 */
struct BlockDocComment;
