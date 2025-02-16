//! This file tests for the `DOC_MARKDOWN` lint, specifically cases
//! where ticks are unbalanced (see issue #6753).
//@no-rustfix
#![allow(dead_code)]
#![warn(clippy::doc_markdown)]

/// This is a doc comment with `unbalanced_tick marks and several words that
//~^ doc_markdown
/// should be `encompassed_by` tick marks because they `contain_underscores`.
/// Because of the initial `unbalanced_tick` pair, the error message is
/// very `confusing_and_misleading`.
fn main() {}

/// This paragraph has `unbalanced_tick marks and should stop_linting.
//~^ doc_markdown
///
/// This paragraph is fine and should_be linted normally.
//~^ doc_markdown
///
/// Double unbalanced backtick from ``here to here` should lint.
//~^ doc_markdown
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
//~^ doc_markdown
///
/// ### `unbalanced
//~^ doc_markdown
///
/// - This `item has unbalanced tick marks
//~^ doc_markdown
/// - This item needs backticks_here
//~^ doc_markdown
fn other_markdown() {}

#[rustfmt::skip]
/// - ```rust
///   /// `lol`
///   pub struct Struct;
///   ```
fn issue_7421() {}

/// `
//~^ doc_markdown
fn escape_0() {}

/// Escaped \` backticks don't count.
fn escape_1() {}

/// Escaped \` \` backticks don't count.
fn escape_2() {}

/// Escaped \` ` backticks don't count, but unescaped backticks do.
//~^ doc_markdown
fn escape_3() {}

/// Backslashes ` \` within code blocks don't count.
fn escape_4() {}
