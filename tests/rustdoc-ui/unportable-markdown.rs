// https://internals.rust-lang.org/t/proposal-migrate-the-syntax-of-rustdoc-markdown-footnotes-to-be-compatible-with-the-syntax-used-in-github/18929
//
// A series of test cases for CommonMark corner cases that pulldown-cmark 0.11 fixes.
//
// This version of the lint is targeted at two especially-common cases where docs got broken.
// Other differences in parsing should not warn.
#![allow(rustdoc::broken_intra_doc_links)]
#![deny(rustdoc::unportable_markdown)]

/// <https://github.com/pulldown-cmark/pulldown-cmark/pull/654>
///
/// Test footnote [^foot].
///
/// [^foot]: This is nested within the footnote now, but didn't used to be.
///
///     This is a multi-paragraph footnote.
pub struct GfmFootnotes;

/// <https://github.com/pulldown-cmark/pulldown-cmark/pull/773>
///
/// test [^foo][^bar]
///
/// [^foo]: test
/// [^bar]: test2
pub struct FootnoteSmashedName;

/// <https://github.com/pulldown-cmark/pulldown-cmark/pull/829>
///
/// - _t
///   # test
///   t_
pub struct NestingCornerCase;

/// <https://github.com/pulldown-cmark/pulldown-cmark/pull/650>
///
/// *~~__emphasis strike strong__~~* ~~*__strike emphasis strong__*~~
pub struct Emphasis1;

/// <https://github.com/pulldown-cmark/pulldown-cmark/pull/732>
///
/// |
/// |
pub struct NotEnoughTable;

/// <https://github.com/pulldown-cmark/pulldown-cmark/pull/675>
///
/// foo
/// >bar
//~^ ERROR unportable markdown
pub struct BlockQuoteNoSpace;

/// Negative test.
///
/// foo
/// > bar
pub struct BlockQuoteSpace;

/// Negative test.
///
/// >bar
/// baz
pub struct BlockQuoteNoSpaceStart;
