use rustc_errors::Applicability;
use rustc_lint::LateContext;
use rustc_resolve::rustdoc::main_body_opts;

use rustc_resolve::rustdoc::pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};

use super::{DOC_PARAGRAPHS_MISSING_PUNCTUATION, Fragments};

const MSG: &str = "doc paragraphs should end with a terminal punctuation mark";
const PUNCTUATION_SUGGESTION: char = '.';

pub fn check(cx: &LateContext<'_>, doc: &str, fragments: Fragments<'_>) {
    for missing_punctuation in is_missing_punctuation(doc) {
        match missing_punctuation {
            MissingPunctuation::Fixable(offset) => {
                // This ignores `#[doc]` attributes, which we do not handle.
                if let Some(span) = fragments.span(cx, offset..offset) {
                    clippy_utils::diagnostics::span_lint_and_sugg(
                        cx,
                        DOC_PARAGRAPHS_MISSING_PUNCTUATION,
                        span,
                        MSG,
                        "end the paragraph with some punctuation",
                        PUNCTUATION_SUGGESTION.to_string(),
                        Applicability::MaybeIncorrect,
                    );
                }
            },
            MissingPunctuation::Unfixable(offset) => {
                // This ignores `#[doc]` attributes, which we do not handle.
                if let Some(span) = fragments.span(cx, offset..offset) {
                    clippy_utils::diagnostics::span_lint_and_help(
                        cx,
                        DOC_PARAGRAPHS_MISSING_PUNCTUATION,
                        span,
                        MSG,
                        None,
                        "end the paragraph with some punctuation",
                    );
                }
            },
        }
    }
}

#[must_use]
/// If punctuation is missing, returns the offset where new punctuation should be inserted.
fn is_missing_punctuation(doc_string: &str) -> Vec<MissingPunctuation> {
    // The colon is not exactly a terminal punctuation mark, but this is required for paragraphs that
    // introduce a table or a list for example.
    const TERMINAL_PUNCTUATION_MARKS: &[char] = &['.', '?', '!', '…', ':'];

    let mut no_report_depth = 0;
    let mut missing_punctuation = Vec::new();
    let mut current_paragraph = None;

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
                current_paragraph = None;
            },
            Event::InlineHtml(_) | Event::Start(Tag::Image { .. }) | Event::End(TagEnd::Image) => {
                current_paragraph = None;
            },
            Event::End(TagEnd::Paragraph) => {
                if let Some(mp) = current_paragraph {
                    missing_punctuation.push(mp);
                }
            },
            Event::Code(..) | Event::Start(Tag::Link { .. }) | Event::End(TagEnd::Link)
                if no_report_depth == 0 && !offset.is_empty() =>
            {
                if doc_string[..offset.end]
                    .trim_end()
                    .ends_with(TERMINAL_PUNCTUATION_MARKS)
                {
                    current_paragraph = None;
                } else {
                    current_paragraph = Some(MissingPunctuation::Fixable(offset.end));
                }
            },
            Event::Text(..) if no_report_depth == 0 && !offset.is_empty() => {
                let trimmed = doc_string[..offset.end].trim_end();
                if trimmed.ends_with(TERMINAL_PUNCTUATION_MARKS) {
                    current_paragraph = None;
                } else if let Some(t) = trimmed.strip_suffix(|c| c == ')' || c == '"') {
                    if t.ends_with(TERMINAL_PUNCTUATION_MARKS) {
                        // Avoid false positives.
                        current_paragraph = None;
                    } else {
                        current_paragraph = Some(MissingPunctuation::Unfixable(offset.end));
                    }
                } else {
                    current_paragraph = Some(MissingPunctuation::Fixable(offset.end));
                }
            },
            _ => {},
        }
    }

    missing_punctuation
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum MissingPunctuation {
    Fixable(usize),
    Unfixable(usize),
}
