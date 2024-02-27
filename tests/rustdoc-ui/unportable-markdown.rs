// https://internals.rust-lang.org/t/proposal-migrate-the-syntax-of-rustdoc-markdown-footnotes-to-be-compatible-with-the-syntax-used-in-github/18929
//
// A series of test cases for CommonMark corner cases that pulldown-cmark 0.10 fixes.

/// <https://github.com/pulldown-cmark/pulldown-cmark/pull/654>
///
/// Test footnote [^foot].
///
/// [^foot]: This is nested within the footnote now, but didn't used to be.
//~^ ERROR unportable markdown
///
///     This is a multi-paragraph footnote.
pub struct GfmFootnotes;

/// <https://github.com/pulldown-cmark/pulldown-cmark/pull/750>
///
/// test [^]
//~^ ERROR unportable markdown
///
/// [^]: test2
pub struct FootnoteEmptyName;

/// <https://github.com/pulldown-cmark/pulldown-cmark/pull/829>
///
/// - _t
///   # test
//~^ ERROR unportable markdown
///   t_
pub struct NestingCornerCase;

/// <https://github.com/pulldown-cmark/pulldown-cmark/pull/650>
///
/// *~~__emphasis strike strong__~~* ~~*__strike emphasis strong__*~~
//~^ ERROR unportable markdown
//~| ERROR unportable markdown
pub struct Emphasis1;

/// <https://github.com/pulldown-cmark/pulldown-cmark/pull/732>
///
/// |
//~^ ERROR unportable markdown
/// |
pub struct NotEnoughTable;
