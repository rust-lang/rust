#![feature(custom_inner_attributes)]
#![rustfmt::skip]
#![warn(clippy::doc_paragraphs_missing_punctuation)]

/// Returns the Answer to the Ultimate Question of Life, the Universe, and Everything
//~^ doc_paragraphs_missing_punctuation
fn answer() -> i32 {
    42
}

/// The `Option` type
//~^ doc_paragraphs_missing_punctuation
// Triggers even in the presence of another attribute.
#[derive(Debug)]
enum MyOption<T> {
    /// No value
    //~^ doc_paragraphs_missing_punctuation
    None,
    /// Some value of type `T`.
    Some(T),
}

// Triggers correctly even when interleaved with other attributes.
/// A multiline
#[derive(Debug)]
/// doc comment:
/// only the last line triggers the lint
//~^ doc_paragraphs_missing_punctuation
enum Exceptions {
    /// Question marks are fine?
    QuestionMark,
    /// Exclamation marks are fine!
    ExclamationMark,
    /// Ellipses are ok too…
    Ellipsis,
    /// HTML content is however not checked:
    /// <em>Raw HTML is allowed as well</em>
    RawHtml,
    /// The raw HTML exception actually does the right thing to autolinks:
    /// <https://spec.commonmark.org/0.31.2/#autolinks>
    //~^ doc_paragraphs_missing_punctuation
    MarkdownAutolink,
    /// This table introduction ends with a colon:
    ///
    /// | Exception      | Note  |
    /// | -------------- | ----- |
    /// | Markdown table | A-ok  |
    MarkdownTable,
    /// Here is a snippet
    //~^ doc_paragraphs_missing_punctuation
    ///
    /// ```
    /// // Code blocks are no issues.
    /// ```
    CodeBlock,
}

// Check the lint can be expected on a whole enum at once.
#[expect(clippy::doc_paragraphs_missing_punctuation)]
enum Char {
    /// U+0000
    Null,
    /// U+0001
    StartOfHeading,
}

// Check the lint can be expected on a single variant without affecting others.
enum Char2 {
    #[expect(clippy::doc_paragraphs_missing_punctuation)]
    /// U+0000
    Null,
    /// U+0001
    //~^ doc_paragraphs_missing_punctuation
    StartOfHeading,
}

mod module {
    //! Works on
    //! inner attributes too
    //~^ doc_paragraphs_missing_punctuation
}

enum Trailers {
    /// Sometimes the last sentence ends with parentheses (and that's ok).
    ParensPassing,
    /// (Sometimes the last sentence is in parentheses.)
    SentenceInParensPassing,
    /// **Sometimes the last sentence is in bold, and that's ok.**
    DoubleStarPassing,
    /// **But sometimes it is missing a period**
    //~^ doc_paragraphs_missing_punctuation
    DoubleStarFailing,
    /// _Sometimes the last sentence is in italics, and that's ok._
    UnderscorePassing,
    /// _But sometimes it is missing a period_
    //~^ doc_paragraphs_missing_punctuation
    UnderscoreFailing,
    /// This comment ends with "a quote."
    AmericanStyleQuotePassing,
    /// This comment ends with "a quote".
    BritishStyleQuotePassing,
}

/// Doc comments can end with an [inline link](#anchor)
//~^ doc_paragraphs_missing_punctuation
struct InlineLink;

/// Some doc comments contain [link reference definitions][spec]
//~^ doc_paragraphs_missing_punctuation
///
/// [spec]: https://spec.commonmark.org/0.31.2/#link-reference-definitions
struct LinkRefDefinition;

// List items do not always need to end with a period.
enum UnorderedLists {
    /// This list has an introductory sentence:
    ///
    /// - A list item
    Dash,
    /// + A list item
    Plus,
    /// * A list item
    Star,
}

enum OrderedLists {
    /// 1. A list item
    Dot,
    /// 42) A list item
    Paren,
}

/// Doc comments with trailing blank lines are supported
//~^ doc_paragraphs_missing_punctuation
///
struct TrailingBlankLine;

/// This doc comment has multiple paragraph.
/// This first paragraph is missing punctuation
//~^ doc_paragraphs_missing_punctuation
///
/// The second one as well
/// And it has multiple sentences
//~^ doc_paragraphs_missing_punctuation
///
/// Same for this third and last one
//~^ doc_paragraphs_missing_punctuation
struct MultiParagraphDocComment;

/// ```
struct IncompleteBlockCode;

/// This ends with a code `span`
//~^ doc_paragraphs_missing_punctuation
struct CodeSpan;

#[expect(clippy::empty_docs)]
///
struct EmptyDocComment;

/**
 * Block doc comments work
 *
 */
//~^^^ doc_paragraphs_missing_punctuation
struct BlockDocComment;

/// Sometimes a doc attribute is used for concatenation
/// ```
#[doc = ""]
/// ```
struct DocAttribute;
