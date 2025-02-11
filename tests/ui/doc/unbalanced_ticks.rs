//! This file tests for the `DOC_MARKDOWN` lint, specifically cases
//! where ticks are unbalanced (see issue #6753).
//@no-rustfix
#![allow(dead_code)]
#![warn(clippy::doc_markdown)]

/// This is a doc comment with `unbalanced_tick marks and several words that
//~^ empty_line_after_doc_comments

/// should be `encompassed_by` tick marks because they `contain_underscores`.
/// Because of the initial `unbalanced_tick` pair, the error message is
/// very `confusing_and_misleading`.
fn main() {}

/// This paragraph has `unbalanced_tick marks and should stop_linting.
//~^ empty_line_after_doc_comments

///
/// This paragraph is fine and should_be linted normally.

///
/// Double unbalanced backtick from ``here to here` should lint.

///
/// Double balanced back ticks ``start end`` is fine.
fn multiple_paragraphs() {}

/// ```
/// // Unbalanced tick mark in code block shouldn't warn:
/// `
/// ```
fn in_code_block() {}

/// # `Fine`
///
/// ## not_fine
//~^ empty_line_after_doc_comments

///
/// ### `unbalanced

///
/// - This `item has unbalanced tick marks

/// - This item needs backticks_here

fn other_markdown() {}

#[rustfmt::skip]
/// - ```rust
///   /// `lol`
///   pub struct Struct;
///   ```
fn issue_7421() {}

/// `
//~^ empty_line_after_doc_comments

fn escape_0() {}

/// Escaped \` backticks don't count.
fn escape_1() {}

/// Escaped \` \` backticks don't count.
fn escape_2() {}

/// Escaped \` ` backticks don't count, but unescaped backticks do.
//~^ empty_line_after_doc_comments

fn escape_3() {}

/// Backslashes ` \` within code blocks don't count.
fn escape_4() {}
