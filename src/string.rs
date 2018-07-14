// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Format string literals.

use regex::Regex;
use unicode_segmentation::UnicodeSegmentation;

use config::Config;
use shape::Shape;
use utils::wrap_str;

const MIN_STRING: usize = 10;

pub struct StringFormat<'a> {
    pub opener: &'a str,
    pub closer: &'a str,
    pub line_start: &'a str,
    pub line_end: &'a str,
    pub shape: Shape,
    pub trim_end: bool,
    pub config: &'a Config,
}

impl<'a> StringFormat<'a> {
    pub fn new(shape: Shape, config: &'a Config) -> StringFormat<'a> {
        StringFormat {
            opener: "\"",
            closer: "\"",
            line_start: " ",
            line_end: "\\",
            shape,
            trim_end: false,
            config,
        }
    }

    /// Returns the maximum number of graphemes that is possible on a line while taking the
    /// indentation into account.
    ///
    /// If we cannot put at least a single character per line, the rewrite won't succeed.
    fn max_chars_with_indent(&self) -> Option<usize> {
        Some(
            self.shape
                .width
                .checked_sub(self.opener.len() + self.line_end.len() + 1)?
                + 1,
        )
    }

    /// Like max_chars_with_indent but the indentation is not substracted.
    /// This allows to fit more graphemes from the string on a line when
    /// SnippetState::Overflow.
    fn max_chars_without_indent(&self) -> Option<usize> {
        Some(self.config.max_width().checked_sub(self.line_end.len())?)
    }
}

pub fn rewrite_string<'a>(orig: &str, fmt: &StringFormat<'a>) -> Option<String> {
    let max_chars_with_indent = fmt.max_chars_with_indent()?;
    let max_chars_without_indent = fmt.max_chars_without_indent()?;
    let indent = fmt.shape.indent.to_string_with_newline(fmt.config);

    // Strip line breaks.
    // With this regex applied, all remaining whitespaces are significant
    let strip_line_breaks_re = Regex::new(r"([^\\](\\\\)*)\\[\n\r][[:space:]]*").unwrap();
    let stripped_str = strip_line_breaks_re.replace_all(orig, "$1");

    let graphemes = UnicodeSegmentation::graphemes(&*stripped_str, false).collect::<Vec<&str>>();

    // `cur_start` is the position in `orig` of the start of the current line.
    let mut cur_start = 0;
    let mut result = String::with_capacity(
        stripped_str
            .len()
            .checked_next_power_of_two()
            .unwrap_or(usize::max_value()),
    );
    result.push_str(fmt.opener);

    // Snip a line at a time from `stripped_str` until it is used up. Push the snippet
    // onto result.
    let mut cur_max_chars = max_chars_with_indent;
    loop {
        // All the input starting at cur_start fits on the current line
        if graphemes.len() - cur_start <= cur_max_chars {
            result.push_str(&graphemes[cur_start..].join(""));
            break;
        }

        // The input starting at cur_start needs to be broken
        match break_string(cur_max_chars, fmt.trim_end, &graphemes[cur_start..]) {
            SnippetState::LineEnd(line, len) => {
                result.push_str(&line);
                result.push_str(fmt.line_end);
                result.push_str(&indent);
                result.push_str(fmt.line_start);
                cur_max_chars = max_chars_with_indent;
                cur_start += len;
            }
            SnippetState::Overflow(line, len) => {
                result.push_str(&line);
                cur_max_chars = max_chars_without_indent;
                cur_start += len;
            }
            SnippetState::EndOfInput(line) => {
                result.push_str(&line);
                break;
            }
        }
    }

    result.push_str(fmt.closer);
    wrap_str(result, fmt.config.max_width(), fmt.shape)
}

/// Result of breaking a string so it fits in a line and the state it ended in.
/// The state informs about what to do with the snippet and how to continue the breaking process.
#[derive(Debug, PartialEq)]
enum SnippetState {
    /// The input could not be broken and so rewriting the string is finished.
    EndOfInput(String),
    /// The input could be broken and the returned snippet should be ended with a
    /// `[StringFormat::line_end]`. The next snippet needs to be indented.
    LineEnd(String, usize),
    /// The input could be broken but the returned snippet should not be ended with a
    /// `[StringFormat::line_end]` because the whitespace is significant. Therefore, the next
    /// snippet should not be indented.
    Overflow(String, usize),
}

/// Break the input string at a boundary character around the offset `max_chars`. A boundary
/// character is either a punctuation or a whitespace.
fn break_string(max_chars: usize, trim_end: bool, input: &[&str]) -> SnippetState {
    let break_at = |index /* grapheme at index is included */| {
        // Take in any whitespaces to the left/right of `input[index]` and
        // check if there is a line feed, in which case whitespaces needs to be kept.
        let mut index_minus_ws = index;
        for (i, grapheme) in input[0..=index].iter().enumerate().rev() {
            if !trim_end && is_line_feed(grapheme) {
                return SnippetState::Overflow(input[0..=i].join("").to_string(), i + 1);
            } else if !is_whitespace(grapheme) {
                index_minus_ws = i;
                break;
            }
        }
        let mut index_plus_ws = index;
        for (i, grapheme) in input[index + 1..].iter().enumerate() {
            if !trim_end && is_line_feed(grapheme) {
                return SnippetState::Overflow(
                    input[0..=index + 1 + i].join("").to_string(),
                    index + 2 + i,
                );
            } else if !is_whitespace(grapheme) {
                index_plus_ws = index + i;
                break;
            }
        }

        if trim_end {
            SnippetState::LineEnd(
                input[0..=index_minus_ws].join("").to_string(),
                index_plus_ws + 1,
            )
        } else {
            SnippetState::LineEnd(
                input[0..=index_plus_ws].join("").to_string(),
                index_plus_ws + 1,
            )
        }
    };

    // Find the position in input for breaking the string
    match input[0..max_chars]
        .iter()
        .rposition(|grapheme| is_whitespace(grapheme))
    {
        // Found a whitespace and what is on its left side is big enough.
        Some(index) if index >= MIN_STRING => break_at(index),
        // No whitespace found, try looking for a punctuation instead
        _ => match input[0..max_chars]
            .iter()
            .rposition(|grapheme| is_punctuation(grapheme))
        {
            // Found a punctuation and what is on its left side is big enough.
            Some(index) if index >= MIN_STRING => break_at(index),
            // Either no boundary character was found to the left of `input[max_chars]`, or the line
            // got too small. We try searching for a boundary character to the right.
            _ => match input[max_chars..]
                .iter()
                .position(|grapheme| is_whitespace(grapheme) || is_punctuation(grapheme))
            {
                // A boundary was found after the line limit
                Some(index) => break_at(max_chars + index),
                // No boundary to the right, the input cannot be broken
                None => SnippetState::EndOfInput(input.join("").to_string()),
            },
        },
    }
}

fn is_line_feed(grapheme: &str) -> bool {
    grapheme.as_bytes()[0] == b'\n'
}

fn is_whitespace(grapheme: &str) -> bool {
    grapheme.chars().all(|c| c.is_whitespace())
}

fn is_punctuation(grapheme: &str) -> bool {
    match grapheme.as_bytes()[0] {
        b':' | b',' | b';' | b'.' => true,
        _ => false,
    }
}

#[cfg(test)]
mod test {
    use super::{break_string, rewrite_string, SnippetState, StringFormat};
    use shape::{Indent, Shape};
    use unicode_segmentation::UnicodeSegmentation;

    #[test]
    fn issue343() {
        let config = Default::default();
        let fmt = StringFormat::new(Shape::legacy(2, Indent::empty()), &config);
        rewrite_string("eq_", &fmt);
    }

    #[test]
    fn should_break_on_whitespace() {
        let string = "Placerat felis. Mauris porta ante sagittis purus.";
        let graphemes = UnicodeSegmentation::graphemes(&*string, false).collect::<Vec<&str>>();
        assert_eq!(
            break_string(20, false, &graphemes[..]),
            SnippetState::LineEnd("Placerat felis. ".to_string(), 16)
        );
        assert_eq!(
            break_string(20, true, &graphemes[..]),
            SnippetState::LineEnd("Placerat felis.".to_string(), 16)
        );
    }

    #[test]
    fn should_break_on_punctuation() {
        let string = "Placerat_felis._Mauris_porta_ante_sagittis_purus.";
        let graphemes = UnicodeSegmentation::graphemes(&*string, false).collect::<Vec<&str>>();
        assert_eq!(
            break_string(20, false, &graphemes[..]),
            SnippetState::LineEnd("Placerat_felis.".to_string(), 15)
        );
    }

    #[test]
    fn should_break_forward() {
        let string = "Venenatis_tellus_vel_tellus. Aliquam aliquam dolor at justo.";
        let graphemes = UnicodeSegmentation::graphemes(&*string, false).collect::<Vec<&str>>();
        assert_eq!(
            break_string(20, false, &graphemes[..]),
            SnippetState::LineEnd("Venenatis_tellus_vel_tellus. ".to_string(), 29)
        );
        assert_eq!(
            break_string(20, true, &graphemes[..]),
            SnippetState::LineEnd("Venenatis_tellus_vel_tellus.".to_string(), 29)
        );
    }

    #[test]
    fn nothing_to_break() {
        let string = "Venenatis_tellus_vel_tellus";
        let graphemes = UnicodeSegmentation::graphemes(&*string, false).collect::<Vec<&str>>();
        assert_eq!(
            break_string(20, false, &graphemes[..]),
            SnippetState::EndOfInput("Venenatis_tellus_vel_tellus".to_string())
        );
    }

    #[test]
    fn significant_whitespaces() {
        let string = "Neque in sem.      \n      Pellentesque tellus augue.";
        let graphemes = UnicodeSegmentation::graphemes(&*string, false).collect::<Vec<&str>>();
        assert_eq!(
            break_string(15, false, &graphemes[..]),
            SnippetState::Overflow("Neque in sem.      \n".to_string(), 20)
        );
        assert_eq!(
            break_string(25, false, &graphemes[..]),
            SnippetState::Overflow("Neque in sem.      \n".to_string(), 20)
        );
        // if `StringFormat::line_end` is true, then the line feed does not matter anymore
        assert_eq!(
            break_string(15, true, &graphemes[..]),
            SnippetState::LineEnd("Neque in sem.".to_string(), 26)
        );
        assert_eq!(
            break_string(25, true, &graphemes[..]),
            SnippetState::LineEnd("Neque in sem.".to_string(), 26)
        );
    }

    #[test]
    fn big_whitespace() {
        let string = "Neque in sem.            Pellentesque tellus augue.";
        let graphemes = UnicodeSegmentation::graphemes(&*string, false).collect::<Vec<&str>>();
        assert_eq!(
            break_string(20, false, &graphemes[..]),
            SnippetState::LineEnd("Neque in sem.            ".to_string(), 25)
        );
        assert_eq!(
            break_string(20, true, &graphemes[..]),
            SnippetState::LineEnd("Neque in sem.".to_string(), 25)
        );
    }
}
