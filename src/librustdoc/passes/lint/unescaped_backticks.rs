//! Detects unescaped backticks (\`) in doc comments.

use std::ops::Range;

use pulldown_cmark::{BrokenLink, Event, Parser};
use rustc_errors::Diag;
use rustc_hir::HirId;
use rustc_lint_defs::Applicability;
use rustc_resolve::rustdoc::source_span_for_markdown_range;

use crate::clean::Item;
use crate::core::DocContext;
use crate::html::markdown::main_body_opts;

pub(crate) fn visit_item(cx: &DocContext<'_>, item: &Item, hir_id: HirId, dox: &str) {
    let tcx = cx.tcx;

    let link_names = item.link_names(&cx.cache);
    let mut replacer = |broken_link: BrokenLink<'_>| {
        link_names
            .iter()
            .find(|link| *link.original_text == *broken_link.reference)
            .map(|link| ((*link.href).into(), (*link.new_text).into()))
    };
    let parser = Parser::new_with_broken_link_callback(dox, main_body_opts(), Some(&mut replacer))
        .into_offset_iter();

    let mut element_stack = Vec::new();

    let mut prev_text_end = 0;
    for (event, event_range) in parser {
        match event {
            Event::Start(_) => {
                element_stack.push(Element::new(event_range));
            }
            Event::End(_) => {
                let element = element_stack.pop().unwrap();

                let Some(backtick_index) = element.backtick_index else {
                    continue;
                };

                // If we can't get a span of the backtick, because it is in a `#[doc = ""]` attribute,
                // use the span of the entire attribute as a fallback.
                let span = match source_span_for_markdown_range(
                    tcx,
                    dox,
                    &(backtick_index..backtick_index + 1),
                    &item.attrs.doc_strings,
                ) {
                    Some((sp, _)) => sp,
                    None => item.attr_span(tcx),
                };

                tcx.node_span_lint(crate::lint::UNESCAPED_BACKTICKS, hir_id, span, |lint| {
                    lint.primary_message("unescaped backtick");

                    let mut help_emitted = false;

                    match element.prev_code_guess {
                        PrevCodeGuess::None => {}
                        PrevCodeGuess::Start { guess, .. } => {
                            // "foo` `bar`" -> "`foo` `bar`"
                            if let Some(suggest_index) =
                                clamp_start(guess, &element.suggestible_ranges)
                                && can_suggest_backtick(dox, suggest_index)
                            {
                                suggest_insertion(
                                    cx,
                                    item,
                                    dox,
                                    lint,
                                    suggest_index,
                                    '`',
                                    "the opening backtick of a previous inline code may be missing",
                                );
                                help_emitted = true;
                            }
                        }
                        PrevCodeGuess::End { guess, .. } => {
                            // "`foo `bar`" -> "`foo` `bar`"
                            // Don't `clamp_end` here, because the suggestion is guaranteed to be inside
                            // an inline code node and we intentionally "break" the inline code here.
                            let suggest_index = guess;
                            if can_suggest_backtick(dox, suggest_index) {
                                suggest_insertion(
                                    cx,
                                    item,
                                    dox,
                                    lint,
                                    suggest_index,
                                    '`',
                                    "a previous inline code might be longer than expected",
                                );
                                help_emitted = true;
                            }
                        }
                    }

                    if !element.prev_code_guess.is_confident() {
                        // "`foo` bar`" -> "`foo` `bar`"
                        if let Some(guess) =
                            guess_start_of_code(dox, element.element_range.start..backtick_index)
                            && let Some(suggest_index) =
                                clamp_start(guess, &element.suggestible_ranges)
                            && can_suggest_backtick(dox, suggest_index)
                        {
                            suggest_insertion(
                                cx,
                                item,
                                dox,
                                lint,
                                suggest_index,
                                '`',
                                "the opening backtick of an inline code may be missing",
                            );
                            help_emitted = true;
                        }

                        // "`foo` `bar" -> "`foo` `bar`"
                        // Don't suggest closing backtick after single trailing char,
                        // if we already suggested opening backtick. For example:
                        // "foo`." -> "`foo`." or "foo`s" -> "`foo`s".
                        if let Some(guess) =
                            guess_end_of_code(dox, backtick_index + 1..element.element_range.end)
                            && let Some(suggest_index) =
                                clamp_end(guess, &element.suggestible_ranges)
                            && can_suggest_backtick(dox, suggest_index)
                            && (!help_emitted || suggest_index - backtick_index > 2)
                        {
                            suggest_insertion(
                                cx,
                                item,
                                dox,
                                lint,
                                suggest_index,
                                '`',
                                "the closing backtick of an inline code may be missing",
                            );
                            help_emitted = true;
                        }
                    }

                    if !help_emitted {
                        lint.help(
                            "the opening or closing backtick of an inline code may be missing",
                        );
                    }

                    suggest_insertion(
                        cx,
                        item,
                        dox,
                        lint,
                        backtick_index,
                        '\\',
                        "if you meant to use a literal backtick, escape it",
                    );
                });
            }
            Event::Code(_) => {
                let element = element_stack
                    .last_mut()
                    .expect("expected inline code node to be inside of an element");
                assert!(
                    event_range.start >= element.element_range.start
                        && event_range.end <= element.element_range.end
                );

                // This inline code might be longer than it's supposed to be.
                // Only check single backtick inline code for now.
                if !element.prev_code_guess.is_confident()
                    && dox.as_bytes().get(event_range.start) == Some(&b'`')
                    && dox.as_bytes().get(event_range.start + 1) != Some(&b'`')
                {
                    let range_inside = event_range.start + 1..event_range.end - 1;
                    let text_inside = &dox[range_inside.clone()];

                    let is_confident = text_inside.starts_with(char::is_whitespace)
                        || text_inside.ends_with(char::is_whitespace);

                    if let Some(guess) = guess_end_of_code(dox, range_inside) {
                        // Find earlier end of code.
                        element.prev_code_guess = PrevCodeGuess::End { guess, is_confident };
                    } else {
                        // Find alternate start of code.
                        let range_before = element.element_range.start..event_range.start;
                        if let Some(guess) = guess_start_of_code(dox, range_before) {
                            element.prev_code_guess = PrevCodeGuess::Start { guess, is_confident };
                        }
                    }
                }
            }
            Event::Text(text) => {
                let element = element_stack
                    .last_mut()
                    .expect("expected inline text node to be inside of an element");
                assert!(
                    event_range.start >= element.element_range.start
                        && event_range.end <= element.element_range.end
                );

                // The first char is escaped if the prev char is \ and not part of a text node.
                let is_escaped = prev_text_end < event_range.start
                    && dox.as_bytes()[event_range.start - 1] == b'\\';

                // Don't lint backslash-escaped (\`) or html-escaped (&#96;) backticks.
                if *text == *"`" && !is_escaped && *text == dox[event_range.clone()] {
                    // We found a stray backtick.
                    assert!(
                        element.backtick_index.is_none(),
                        "expected at most one unescaped backtick per element",
                    );
                    element.backtick_index = Some(event_range.start);
                }

                prev_text_end = event_range.end;

                if is_escaped {
                    // Ensure that we suggest "`\x" and not "\`x".
                    element.suggestible_ranges.push(event_range.start - 1..event_range.end);
                } else {
                    element.suggestible_ranges.push(event_range);
                }
            }
            _ => {}
        }
    }
}

/// A previous inline code node, that looks wrong.
///
/// `guess` is the position, where we want to suggest a \` and the guess `is_confident` if an
/// inline code starts or ends with a whitespace.
#[derive(Debug)]
enum PrevCodeGuess {
    None,

    /// Missing \` at start.
    ///
    /// ```markdown
    /// foo` `bar`
    /// ```
    Start {
        guess: usize,
        is_confident: bool,
    },

    /// Missing \` at end.
    ///
    /// ```markdown
    /// `foo `bar`
    /// ```
    End {
        guess: usize,
        is_confident: bool,
    },
}

impl PrevCodeGuess {
    fn is_confident(&self) -> bool {
        match *self {
            PrevCodeGuess::None => false,
            PrevCodeGuess::Start { is_confident, .. } | PrevCodeGuess::End { is_confident, .. } => {
                is_confident
            }
        }
    }
}

/// A markdown [tagged element], which may or may not contain an unescaped backtick.
///
/// [tagged element]: https://docs.rs/pulldown-cmark/0.9/pulldown_cmark/enum.Tag.html
#[derive(Debug)]
struct Element {
    /// The full range (span) of the element in the doc string.
    element_range: Range<usize>,

    /// The ranges where we're allowed to put backticks.
    /// This is used to prevent breaking markdown elements like links or lists.
    suggestible_ranges: Vec<Range<usize>>,

    /// The unescaped backtick.
    backtick_index: Option<usize>,

    /// Suggest a different start or end of an inline code.
    prev_code_guess: PrevCodeGuess,
}

impl Element {
    const fn new(element_range: Range<usize>) -> Self {
        Self {
            element_range,
            suggestible_ranges: Vec::new(),
            backtick_index: None,
            prev_code_guess: PrevCodeGuess::None,
        }
    }
}

/// Given a potentially unclosed inline code, attempt to find the start.
fn guess_start_of_code(dox: &str, range: Range<usize>) -> Option<usize> {
    assert!(dox.as_bytes()[range.end] == b'`');

    let mut braces = 0;
    let mut guess = 0;
    for (idx, ch) in dox[range.clone()].char_indices().rev() {
        match ch {
            ')' | ']' | '}' => braces += 1,
            '(' | '[' | '{' => {
                if braces == 0 {
                    guess = idx + 1;
                    break;
                }
                braces -= 1;
            }
            ch if ch.is_whitespace() && braces == 0 => {
                guess = idx + 1;
                break;
            }
            _ => (),
        }
    }

    guess += range.start;

    // Don't suggest empty inline code or duplicate backticks.
    can_suggest_backtick(dox, guess).then_some(guess)
}

/// Given a potentially unclosed inline code, attempt to find the end.
fn guess_end_of_code(dox: &str, range: Range<usize>) -> Option<usize> {
    // Punctuation that should be outside of the inline code.
    const TRAILING_PUNCTUATION: &[u8] = b".,";

    assert!(dox.as_bytes()[range.start - 1] == b'`');

    let text = dox[range.clone()].trim_end();
    let mut braces = 0;
    let mut guess = text.len();
    for (idx, ch) in text.char_indices() {
        match ch {
            '(' | '[' | '{' => braces += 1,
            ')' | ']' | '}' => {
                if braces == 0 {
                    guess = idx;
                    break;
                }
                braces -= 1;
            }
            ch if ch.is_whitespace() && braces == 0 => {
                guess = idx;
                break;
            }
            _ => (),
        }
    }

    // Strip a single trailing punctuation.
    if guess >= 1
        && TRAILING_PUNCTUATION.contains(&text.as_bytes()[guess - 1])
        && (guess < 2 || !TRAILING_PUNCTUATION.contains(&text.as_bytes()[guess - 2]))
    {
        guess -= 1;
    }

    guess += range.start;

    // Don't suggest empty inline code or duplicate backticks.
    can_suggest_backtick(dox, guess).then_some(guess)
}

/// Returns whether inserting a backtick at `dox[index]` will not produce double backticks.
fn can_suggest_backtick(dox: &str, index: usize) -> bool {
    (index == 0 || dox.as_bytes()[index - 1] != b'`')
        && (index == dox.len() || dox.as_bytes()[index] != b'`')
}

/// Increase the index until it is inside or one past the end of one of the ranges.
///
/// The ranges must be sorted for this to work correctly.
fn clamp_start(index: usize, ranges: &[Range<usize>]) -> Option<usize> {
    for range in ranges {
        if range.start >= index {
            return Some(range.start);
        }
        if index <= range.end {
            return Some(index);
        }
    }
    None
}

/// Decrease the index until it is inside or one past the end of one of the ranges.
///
/// The ranges must be sorted for this to work correctly.
fn clamp_end(index: usize, ranges: &[Range<usize>]) -> Option<usize> {
    for range in ranges.iter().rev() {
        if range.end <= index {
            return Some(range.end);
        }
        if index >= range.start {
            return Some(index);
        }
    }
    None
}

/// Try to emit a span suggestion and fall back to help messages if we can't find a suitable span.
///
/// This helps finding backticks in huge macro-generated docs.
fn suggest_insertion(
    cx: &DocContext<'_>,
    item: &Item,
    dox: &str,
    lint: &mut Diag<'_, ()>,
    insert_index: usize,
    suggestion: char,
    message: &'static str,
) {
    /// Maximum bytes of context to show around the insertion.
    const CONTEXT_MAX_LEN: usize = 80;

    if let Some((span, _)) = source_span_for_markdown_range(
        cx.tcx,
        dox,
        &(insert_index..insert_index),
        &item.attrs.doc_strings,
    ) {
        lint.span_suggestion(span, message, suggestion, Applicability::MaybeIncorrect);
    } else {
        let line_start = dox[..insert_index].rfind('\n').map_or(0, |idx| idx + 1);
        let line_end = dox[insert_index..].find('\n').map_or(dox.len(), |idx| idx + insert_index);

        let context_before_max_len = if insert_index - line_start < CONTEXT_MAX_LEN / 2 {
            insert_index - line_start
        } else if line_end - insert_index < CONTEXT_MAX_LEN / 2 {
            CONTEXT_MAX_LEN - (line_end - insert_index)
        } else {
            CONTEXT_MAX_LEN / 2
        };
        let context_after_max_len = CONTEXT_MAX_LEN - context_before_max_len;

        let (prefix, context_start) = if insert_index - line_start <= context_before_max_len {
            ("", line_start)
        } else {
            ("...", dox.ceil_char_boundary(insert_index - context_before_max_len))
        };
        let (suffix, context_end) = if line_end - insert_index <= context_after_max_len {
            ("", line_end)
        } else {
            ("...", dox.floor_char_boundary(insert_index + context_after_max_len))
        };

        let context_full = &dox[context_start..context_end].trim_end();
        let context_before = &dox[context_start..insert_index];
        let context_after = &dox[insert_index..context_end].trim_end();
        lint.help(format!(
            "{message}\n change: {prefix}{context_full}{suffix}\nto this: {prefix}{context_before}{suggestion}{context_after}{suffix}"
        ));
    }
}
