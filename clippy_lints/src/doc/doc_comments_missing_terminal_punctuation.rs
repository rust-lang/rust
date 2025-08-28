use rustc_ast::ast::{AttrKind, AttrStyle, Attribute};
use rustc_errors::Applicability;
use rustc_lint::EarlyContext;

use super::DOC_COMMENTS_MISSING_TERMINAL_PUNCTUATION;

const MSG: &str = "doc comments should end with a terminal punctuation mark";
const PUNCTUATION_SUGGESTION: char = '.';

pub fn check(cx: &EarlyContext<'_>, attrs: &[Attribute]) {
    let mut doc_comment_attrs = attrs.iter().enumerate().filter(|(_, a)| is_doc_comment(a));

    let Some((i, mut last_doc_attr)) = doc_comment_attrs.next_back() else {
        return;
    };

    // Check that the next attribute is not a `#[doc]` attribute.
    if let Some(next_attr) = attrs.get(i + 1)
        && is_doc_attr(next_attr)
    {
        return;
    }

    // Find the last, non-blank, non-refdef line of multiline doc comments: this is enough to check that
    // the doc comment ends with proper punctuation.
    while is_doc_comment_trailer(last_doc_attr) {
        if let Some(doc_attr) = doc_comment_attrs.next_back() {
            (_, last_doc_attr) = doc_attr;
        } else {
            // The doc comment looks (functionally) empty.
            return;
        }
    }

    if let Some(doc_string) = is_missing_punctuation(last_doc_attr) {
        let span = last_doc_attr.span;

        if is_line_doc_comment(last_doc_attr) {
            let suggestion = generate_suggestion(last_doc_attr, doc_string);

            clippy_utils::diagnostics::span_lint_and_sugg(
                cx,
                DOC_COMMENTS_MISSING_TERMINAL_PUNCTUATION,
                span,
                MSG,
                "end the doc comment with some punctuation",
                suggestion,
                Applicability::MaybeIncorrect,
            );
        } else {
            // Seems more difficult to preserve the formatting of block doc comments, so we do not provide
            // suggestions for them; they are much rarer anyway.
            clippy_utils::diagnostics::span_lint(cx, DOC_COMMENTS_MISSING_TERMINAL_PUNCTUATION, span, MSG);
        }
    }
}

#[must_use]
fn is_missing_punctuation(attr: &Attribute) -> Option<&str> {
    const TERMINAL_PUNCTUATION_MARKS: &[char] = &['.', '?', '!', '…'];
    const EXCEPTIONS: &[char] = &[
        '>', // Raw HTML or (unfortunately) Markdown autolinks.
        '|', // Markdown tables.
    ];

    let doc_string = get_doc_string(attr)?;

    // Doc comments could have some trailing whitespace, but that is not this lint's job.
    let trimmed = doc_string.trim_end();

    // Doc comments are also allowed to end with fenced code blocks.
    if trimmed.ends_with(TERMINAL_PUNCTUATION_MARKS) || trimmed.ends_with(EXCEPTIONS) || trimmed.ends_with("```") {
        return None;
    }

    // Ignore single-line list items: they may not require any terminal punctuation.
    if looks_like_list_item(trimmed) {
        return None;
    }

    if let Some(stripped) = strip_sentence_trailers(trimmed)
        && stripped.ends_with(TERMINAL_PUNCTUATION_MARKS)
    {
        return None;
    }

    Some(doc_string)
}

#[must_use]
fn generate_suggestion(doc_attr: &Attribute, doc_string: &str) -> String {
    let doc_comment_prefix = match doc_attr.style {
        AttrStyle::Outer => "///",
        AttrStyle::Inner => "//!",
    };

    let mut original_line = format!("{doc_comment_prefix}{doc_string}");

    if let Some(stripped) = strip_sentence_trailers(doc_string) {
        // Insert the punctuation mark just before the sentence trailer.
        original_line.insert(doc_comment_prefix.len() + stripped.len(), PUNCTUATION_SUGGESTION);
    } else {
        original_line.push(PUNCTUATION_SUGGESTION);
    }

    original_line
}

/// Strips closing parentheses and Markdown emphasis delimiters.
#[must_use]
fn strip_sentence_trailers(string: &str) -> Option<&str> {
    // The std has a few occurrences of doc comments ending with a sentence in parentheses.
    const TRAILERS: &[char] = &[')', '*', '_'];

    if let Some(stripped) = string.strip_suffix("**") {
        return Some(stripped);
    }

    if let Some(stripped) = string.strip_suffix("__") {
        return Some(stripped);
    }

    // Markdown inline links should not be mistaken for sentences in parentheses.
    if looks_like_inline_link(string) {
        return None;
    }

    string.strip_suffix(TRAILERS)
}

/// Returns whether the doc comment looks like a Markdown reference definition or a blank line.
#[must_use]
fn is_doc_comment_trailer(attr: &Attribute) -> bool {
    let Some(doc_string) = get_doc_string(attr) else {
        return false;
    };

    super::looks_like_refdef(doc_string, 0..doc_string.len()).is_some() || doc_string.trim_end().is_empty()
}

/// Returns whether the string looks like it ends with a Markdown inline link.
#[must_use]
fn looks_like_inline_link(string: &str) -> bool {
    let Some(sub) = string.strip_suffix(')') else {
        return false;
    };
    let Some((sub, _)) = sub.rsplit_once('(') else {
        return false;
    };

    // Check whether there is closing bracket just before the opening parenthesis.
    sub.ends_with(']')
}

/// Returns whether the string looks like a Markdown list item.
#[must_use]
fn looks_like_list_item(string: &str) -> bool {
    const BULLET_LIST_MARKERS: &[char] = &['-', '+', '*'];
    const ORDERED_LIST_MARKER_SYMBOL: &[char] = &['.', ')'];

    let trimmed = string.trim_start();

    if let Some(sub) = trimmed.strip_prefix(BULLET_LIST_MARKERS)
        && sub.starts_with(char::is_whitespace)
    {
        return true;
    }

    let mut stripped = trimmed;
    while let Some(sub) = stripped.strip_prefix(|c| char::is_digit(c, 10)) {
        stripped = sub;
    }
    if let Some(sub) = stripped.strip_prefix(ORDERED_LIST_MARKER_SYMBOL)
        && sub.starts_with(char::is_whitespace)
    {
        return true;
    }

    false
}

#[must_use]
fn is_doc_attr(attr: &Attribute) -> bool {
    if let AttrKind::Normal(normal_attr) = &attr.kind
        && let Some(segment) = &normal_attr.item.path.segments.first()
        && segment.ident.name == clippy_utils::sym::doc
    {
        true
    } else {
        false
    }
}

#[must_use]
fn get_doc_string(attr: &Attribute) -> Option<&str> {
    if let AttrKind::DocComment(_, symbol) = &attr.kind {
        Some(symbol.as_str())
    } else {
        None
    }
}

#[must_use]
fn is_doc_comment(attr: &Attribute) -> bool {
    matches!(attr.kind, AttrKind::DocComment(_, _))
}

#[must_use]
fn is_line_doc_comment(attr: &Attribute) -> bool {
    matches!(attr.kind, AttrKind::DocComment(rustc_ast::token::CommentKind::Line, _))
}
