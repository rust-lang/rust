use rustc_errors::Applicability;
use rustc_lint::LateContext;
use rustc_resolve::rustdoc::main_body_opts;

use rustc_resolve::rustdoc::pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};

use super::{DOC_COMMENTS_MISSING_TERMINAL_PUNCTUATION, Fragments};

const MSG: &str = "doc comments should end with a terminal punctuation mark";
const PUNCTUATION_SUGGESTION: char = '.';

pub fn check(cx: &LateContext<'_>, doc: &str, fragments: Fragments<'_>) {
    // This ignores `#[doc]` attributes, which we do not handle.
    if let Some(offset) = is_missing_punctuation(doc)
        && let Some(span) = fragments.span(cx, offset..offset)
    {
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
}

#[must_use]
/// If punctuation is missing, returns the offset where new punctuation should be inserted.
fn is_missing_punctuation(doc_string: &str) -> Option<usize> {
    const TERMINAL_PUNCTUATION_MARKS: &[char] = &['.', '?', '!', '…'];

    let mut no_report_depth = 0;
    let mut text_offset = None;
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
                text_offset = None;
            },
            Event::InlineHtml(_) | Event::Start(Tag::Image { .. }) | Event::End(TagEnd::Image) => {
                text_offset = None;
            },
            Event::Code(..) | Event::Text(..) | Event::Start(Tag::Link { .. }) | Event::End(TagEnd::Link)
                if no_report_depth == 0 && !offset.is_empty() =>
            {
                text_offset = Some(offset.end);
            },
            _ => {},
        }
    }

    let text_offset = text_offset?;
    if doc_string[..text_offset]
        .trim_end()
        .ends_with(TERMINAL_PUNCTUATION_MARKS)
    {
        None
    } else {
        Some(text_offset)
    }
}
