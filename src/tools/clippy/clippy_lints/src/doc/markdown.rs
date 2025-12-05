use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::snippet_with_applicability;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_lint::LateContext;
use rustc_span::{BytePos, Pos, Span};
use url::Url;

use crate::doc::{DOC_MARKDOWN, Fragments};
use std::ops::Range;

pub fn check(
    cx: &LateContext<'_>,
    valid_idents: &FxHashSet<String>,
    text: &str,
    fragments: &Fragments<'_>,
    fragment_range: Range<usize>,
    code_level: isize,
    blockquote_level: isize,
) {
    for orig_word in text.split(|c: char| c.is_whitespace() || c == '\'') {
        // Trim punctuation as in `some comment (see foo::bar).`
        //                                                   ^^
        // Or even as in `_foo bar_` which is emphasized. Also preserve `::` as a prefix/suffix.
        let trim_pattern = |c: char| !c.is_alphanumeric() && c != ':';
        let mut word = orig_word.trim_end_matches(trim_pattern);

        // If word is immediately followed by `()`, claw it back.
        if let Some(tmp_word) = orig_word.get(..word.len() + 2)
            && tmp_word.ends_with("()")
        {
            word = tmp_word;
        }

        let original_len = word.len();
        word = word.trim_start_matches(trim_pattern);

        // Remove leading or trailing single `:` which may be part of a sentence.
        if word.starts_with(':') && !word.starts_with("::") {
            word = word.trim_start_matches(':');
        }
        if word.ends_with(':') && !word.ends_with("::") {
            word = word.trim_end_matches(':');
        }

        if valid_idents.contains(word) || word.chars().all(|c| c == ':') {
            continue;
        }

        // Ensure that all reachable matching closing parens are included as well.
        let size_diff = original_len - word.len();
        let mut open_parens = 0;
        let mut close_parens = 0;
        for c in word.chars() {
            if c == '(' {
                open_parens += 1;
            } else if c == ')' {
                close_parens += 1;
            }
        }
        while close_parens < open_parens
            && let Some(tmp_word) = orig_word.get(size_diff..=(word.len() + size_diff))
            && tmp_word.ends_with(')')
        {
            word = tmp_word;
            close_parens += 1;
        }

        // We'll use this offset to calculate the span to lint.
        let fragment_offset = word.as_ptr() as usize - text.as_ptr() as usize;

        // Adjust for the current word
        check_word(
            cx,
            word,
            fragments,
            &fragment_range,
            fragment_offset,
            code_level,
            blockquote_level,
        );
    }
}

fn check_word(
    cx: &LateContext<'_>,
    word: &str,
    fragments: &Fragments<'_>,
    range: &Range<usize>,
    fragment_offset: usize,
    code_level: isize,
    blockquote_level: isize,
) {
    /// Checks if a string is upper-camel-case, i.e., starts with an uppercase and
    /// contains at least two uppercase letters (`Clippy` is ok) and one lower-case
    /// letter (`NASA` is ok).
    /// Plurals are also excluded (`IDs` is ok).
    fn is_camel_case(s: &str) -> bool {
        if s.starts_with(|c: char| c.is_ascii_digit() | c.is_ascii_lowercase()) {
            return false;
        }

        let s = if let Some(prefix) = s.strip_suffix("es")
            && prefix.chars().all(|c| c.is_ascii_uppercase())
            && matches!(prefix.chars().last(), Some('S' | 'X'))
        {
            prefix
        } else if let Some(prefix) = s.strip_suffix("ified")
            && prefix.chars().all(|c| c.is_ascii_uppercase())
        {
            prefix
        } else {
            s.strip_suffix('s').unwrap_or(s)
        };

        s.chars().all(char::is_alphanumeric)
            && s.chars().filter(|&c| c.is_uppercase()).take(2).count() > 1
            && s.chars().filter(|&c| c.is_lowercase()).take(1).count() > 0
    }

    fn has_underscore(s: &str) -> bool {
        s != "_" && !s.contains("\\_") && s.contains('_')
    }

    fn has_hyphen(s: &str) -> bool {
        s != "-" && s.contains('-')
    }

    if let Ok(url) = Url::parse(word)
        // try to get around the fact that `foo::bar` parses as a valid URL
        && !url.cannot_be_a_base()
    {
        let Some(fragment_span) = fragments.span(cx, range.clone()) else {
            return;
        };
        let span = Span::new(
            fragment_span.lo() + BytePos::from_usize(fragment_offset),
            fragment_span.lo() + BytePos::from_usize(fragment_offset + word.len()),
            fragment_span.ctxt(),
            fragment_span.parent(),
        );

        span_lint_and_sugg(
            cx,
            DOC_MARKDOWN,
            span,
            "you should put bare URLs between `<`/`>` or make a proper Markdown link",
            "try",
            format!("<{word}>"),
            Applicability::MachineApplicable,
        );
        return;
    }

    // We assume that mixed-case words are not meant to be put inside backticks. (Issue #2343)
    //
    // We also assume that backticks are not necessary if inside a quote. (Issue #10262)
    if code_level > 0 || blockquote_level > 0 || (has_underscore(word) && has_hyphen(word)) {
        return;
    }

    if has_underscore(word) || word.contains("::") || is_camel_case(word) || word.ends_with("()") {
        let Some(fragment_span) = fragments.span(cx, range.clone()) else {
            return;
        };

        let span = Span::new(
            fragment_span.lo() + BytePos::from_usize(fragment_offset),
            fragment_span.lo() + BytePos::from_usize(fragment_offset + word.len()),
            fragment_span.ctxt(),
            fragment_span.parent(),
        );

        span_lint_and_then(
            cx,
            DOC_MARKDOWN,
            span,
            "item in documentation is missing backticks",
            |diag| {
                let mut applicability = Applicability::MachineApplicable;
                let snippet = snippet_with_applicability(cx, span, "..", &mut applicability);
                diag.span_suggestion_verbose(span, "try", format!("`{snippet}`"), applicability);
            },
        );
    }
}
