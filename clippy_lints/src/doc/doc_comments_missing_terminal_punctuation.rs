use rustc_errors::Applicability;
use rustc_lint::LateContext;
use rustc_resolve::rustdoc::main_body_opts;

use rustc_resolve::rustdoc::pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};

use super::{DOC_COMMENTS_MISSING_TERMINAL_PUNCTUATION, Fragments};

const MSG: &str = "doc comments should end with a terminal punctuation mark";
const PUNCTUATION_SUGGESTION: char = '.';

pub fn check(cx: &LateContext<'_>, doc: &str, fragments: Fragments<'_>) {
    match is_missing_punctuation(doc) {
        IsMissingPunctuation::Fixable(offset) => {
            // This ignores `#[doc]` attributes, which we do not handle.
            if let Some(span) = fragments.span(cx, offset..offset) {
                clippy_utils::diagnostics::span_lint_and_sugg(
                    cx,
                    DOC_COMMENTS_MISSING_TERMINAL_PUNCTUATION,
                    span,
                    MSG,
                    "end the doc comment with some punctuation",
                    PUNCTUATION_SUGGESTION.to_string(),
                    Applicability::MaybeIncorrect,
                );
            }
        },
        IsMissingPunctuation::Unfixable(offset) => {
            // This ignores `#[doc]` attributes, which we do not handle.
            if let Some(span) = fragments.span(cx, offset..offset) {
                clippy_utils::diagnostics::span_lint_and_help(
                    cx,
                    DOC_COMMENTS_MISSING_TERMINAL_PUNCTUATION,
                    span,
                    MSG,
                    None,
                    "end the doc comment with some punctuation",
                );
            }
        },
        IsMissingPunctuation::No => {},
    }
}

#[must_use]
/// If punctuation is missing, returns the offset where new punctuation should be inserted.
fn is_missing_punctuation(doc_string: &str) -> IsMissingPunctuation {
    const TERMINAL_PUNCTUATION_MARKS: &[char] = &['.', '?', '!', '…'];

    // Short-circuit in simple, common cases to avoid Markdown parsing.
    if doc_string.trim_end().ends_with(TERMINAL_PUNCTUATION_MARKS) {
        return IsMissingPunctuation::No;
    }

    let mut no_report_depth = 0;
    let mut missing_punctuation = IsMissingPunctuation::No;
    for (event, offset) in
        Parser::new_ext(doc_string, main_body_opts() - Options::ENABLE_SMART_PUNCTUATION).into_offset_iter()
    {
        match event {
            Event::Start(
                Tag::CodeBlock(..)
                | Tag::FootnoteDefinition(_)
                | Tag::Heading { .. }
                | Tag::HtmlBlock
                | Tag::List(..)
                | Tag::Table(_),
            ) => {
                no_report_depth += 1;
            },
            Event::End(TagEnd::FootnoteDefinition) => {
                no_report_depth -= 1;
            },
            Event::End(
                TagEnd::CodeBlock | TagEnd::Heading(_) | TagEnd::HtmlBlock | TagEnd::List(_) | TagEnd::Table,
            ) => {
                no_report_depth -= 1;
                missing_punctuation = IsMissingPunctuation::No;
            },
            Event::InlineHtml(_) | Event::Start(Tag::Image { .. }) | Event::End(TagEnd::Image) => {
                missing_punctuation = IsMissingPunctuation::No;
            },
            Event::Code(..) | Event::Start(Tag::Link { .. }) | Event::End(TagEnd::Link)
                if no_report_depth == 0 && !offset.is_empty() =>
            {
                if doc_string[..offset.end]
                    .trim_end()
                    .ends_with(TERMINAL_PUNCTUATION_MARKS)
                {
                    missing_punctuation = IsMissingPunctuation::No;
                } else {
                    missing_punctuation = IsMissingPunctuation::Fixable(offset.end);
                }
            },
            Event::Text(..) if no_report_depth == 0 && !offset.is_empty() => {
                let trimmed = doc_string[..offset.end].trim_end();
                if trimmed.ends_with(TERMINAL_PUNCTUATION_MARKS) {
                    missing_punctuation = IsMissingPunctuation::No;
                } else if let Some(t) = trimmed.strip_suffix(|c| c == ')' || c == '"') {
                    if t.ends_with(TERMINAL_PUNCTUATION_MARKS) {
                        // Avoid false positives.
                        missing_punctuation = IsMissingPunctuation::No;
                    } else {
                        missing_punctuation = IsMissingPunctuation::Unfixable(offset.end);
                    }
                } else {
                    missing_punctuation = IsMissingPunctuation::Fixable(offset.end);
                }
            },
            _ => {},
        }
    }

    missing_punctuation
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum IsMissingPunctuation {
    Fixable(usize),
    Unfixable(usize),
    No,
}
