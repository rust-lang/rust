// Format string literals.

use regex::Regex;
use unicode_properties::{GeneralCategory, UnicodeGeneralCategory};
use unicode_segmentation::UnicodeSegmentation;

use crate::config::Config;
use crate::shape::Shape;
use crate::utils::{unicode_str_width, wrap_str};

const MIN_STRING: usize = 10;

/// Describes the layout of a piece of text.
pub(crate) struct StringFormat<'a> {
    /// The opening sequence of characters for the piece of text
    pub(crate) opener: &'a str,
    /// The closing sequence of characters for the piece of text
    pub(crate) closer: &'a str,
    /// The opening sequence of characters for a line
    pub(crate) line_start: &'a str,
    /// The closing sequence of characters for a line
    pub(crate) line_end: &'a str,
    /// The allocated box to fit the text into
    pub(crate) shape: Shape,
    /// Trim trailing whitespaces
    pub(crate) trim_end: bool,
    pub(crate) config: &'a Config,
}

impl<'a> StringFormat<'a> {
    pub(crate) fn new(shape: Shape, config: &'a Config) -> StringFormat<'a> {
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
    fn max_width_with_indent(&self) -> Option<usize> {
        Some(
            self.shape
                .width
                .checked_sub(self.opener.len() + self.line_end.len() + 1)?
                + 1,
        )
    }

    /// Like max_width_with_indent but the indentation is not subtracted.
    /// This allows to fit more graphemes from the string on a line when
    /// SnippetState::EndWithLineFeed.
    fn max_width_without_indent(&self) -> Option<usize> {
        self.config.max_width().checked_sub(self.line_end.len())
    }
}

pub(crate) fn rewrite_string<'a>(
    orig: &str,
    fmt: &StringFormat<'a>,
    newline_max_chars: usize,
) -> Option<String> {
    let max_width_with_indent = fmt.max_width_with_indent()?;
    let max_width_without_indent = fmt.max_width_without_indent()?;
    let indent_with_newline = fmt.shape.indent.to_string_with_newline(fmt.config);
    let indent_without_newline = fmt.shape.indent.to_string(fmt.config);

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
            .unwrap_or(usize::MAX),
    );
    result.push_str(fmt.opener);

    // Snip a line at a time from `stripped_str` until it is used up. Push the snippet
    // onto result.
    let mut cur_max_width = max_width_with_indent;
    let is_bareline_ok = fmt.line_start.is_empty() || is_whitespace(fmt.line_start);
    loop {
        // All the input starting at cur_start fits on the current line
        if graphemes_width(&graphemes[cur_start..]) <= cur_max_width {
            for (i, grapheme) in graphemes[cur_start..].iter().enumerate() {
                if is_new_line(grapheme) {
                    // take care of blank lines
                    result = trim_end_but_line_feed(fmt.trim_end, result);
                    result.push('\n');
                    if !is_bareline_ok && cur_start + i + 1 < graphemes.len() {
                        result.push_str(&indent_without_newline);
                        result.push_str(fmt.line_start);
                    }
                } else {
                    result.push_str(grapheme);
                }
            }
            result = trim_end_but_line_feed(fmt.trim_end, result);
            break;
        }

        // The input starting at cur_start needs to be broken
        match break_string(
            cur_max_width,
            fmt.trim_end,
            fmt.line_end,
            &graphemes[cur_start..],
        ) {
            SnippetState::LineEnd(line, len) => {
                result.push_str(&line);
                result.push_str(fmt.line_end);
                result.push_str(&indent_with_newline);
                result.push_str(fmt.line_start);
                cur_max_width = newline_max_chars;
                cur_start += len;
            }
            SnippetState::EndWithLineFeed(line, len) => {
                if line == "\n" && fmt.trim_end {
                    result = result.trim_end().to_string();
                }
                result.push_str(&line);
                if is_bareline_ok {
                    // the next line can benefit from the full width
                    cur_max_width = max_width_without_indent;
                } else {
                    result.push_str(&indent_without_newline);
                    result.push_str(fmt.line_start);
                    cur_max_width = max_width_with_indent;
                }
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

/// Returns the index to the end of the URL if the split at index of the given string includes a
/// URL or alike. Otherwise, returns `None`.
fn detect_url(s: &[&str], index: usize) -> Option<usize> {
    let start = match s[..=index].iter().rposition(|g| is_whitespace(g)) {
        Some(pos) => pos + 1,
        None => 0,
    };
    // 8 = minimum length for a string to contain a URL
    if s.len() < start + 8 {
        return None;
    }
    let split = s[start..].concat();
    if split.contains("https://")
        || split.contains("http://")
        || split.contains("ftp://")
        || split.contains("file://")
    {
        match s[index..].iter().position(|g| is_whitespace(g)) {
            Some(pos) => Some(index + pos - 1),
            None => Some(s.len() - 1),
        }
    } else {
        None
    }
}

/// Trims whitespaces to the right except for the line feed character.
fn trim_end_but_line_feed(trim_end: bool, result: String) -> String {
    let whitespace_except_line_feed = |c: char| c.is_whitespace() && c != '\n';
    if trim_end && result.ends_with(whitespace_except_line_feed) {
        result
            .trim_end_matches(whitespace_except_line_feed)
            .to_string()
    } else {
        result
    }
}

/// Result of breaking a string so it fits in a line and the state it ended in.
/// The state informs about what to do with the snippet and how to continue the breaking process.
#[derive(Debug, PartialEq)]
enum SnippetState {
    /// The input could not be broken and so rewriting the string is finished.
    EndOfInput(String),
    /// The input could be broken and the returned snippet should be ended with a
    /// `[StringFormat::line_end]`. The next snippet needs to be indented.
    ///
    /// The returned string is the line to print out and the number is the length that got read in
    /// the text being rewritten. That length may be greater than the returned string if trailing
    /// whitespaces got trimmed.
    LineEnd(String, usize),
    /// The input could be broken but a newline is present that cannot be trimmed. The next snippet
    /// to be rewritten *could* use more width than what is specified by the given shape. For
    /// example with a multiline string, the next snippet does not need to be indented, allowing
    /// more characters to be fit within a line.
    ///
    /// The returned string is the line to print out and the number is the length that got read in
    /// the text being rewritten.
    EndWithLineFeed(String, usize),
}

fn not_whitespace_except_line_feed(g: &str) -> bool {
    is_new_line(g) || !is_whitespace(g)
}

/// Break the input string at a boundary character around the offset `max_width`. A boundary
/// character is either a punctuation or a whitespace.
/// FIXME(issue#3281): We must follow UAX#14 algorithm instead of this.
fn break_string(max_width: usize, trim_end: bool, line_end: &str, input: &[&str]) -> SnippetState {
    let break_at = |index /* grapheme at index is included */| {
        // Take in any whitespaces to the left/right of `input[index]` while
        // preserving line feeds
        let index_minus_ws = input[0..=index]
            .iter()
            .rposition(|grapheme| not_whitespace_except_line_feed(grapheme))
            .unwrap_or(index);
        // Take into account newlines occurring in input[0..=index], i.e., the possible next new
        // line. If there is one, then text after it could be rewritten in a way that the available
        // space is fully used.
        for (i, grapheme) in input[0..=index].iter().enumerate() {
            if is_new_line(grapheme) {
                if i <= index_minus_ws {
                    let mut line = &input[0..i].concat()[..];
                    if trim_end {
                        line = line.trim_end();
                    }
                    return SnippetState::EndWithLineFeed(format!("{}\n", line), i + 1);
                }
                break;
            }
        }

        let mut index_plus_ws = index;
        for (i, grapheme) in input[index + 1..].iter().enumerate() {
            if !trim_end && is_new_line(grapheme) {
                return SnippetState::EndWithLineFeed(
                    input[0..=index + 1 + i].concat(),
                    index + 2 + i,
                );
            } else if not_whitespace_except_line_feed(grapheme) {
                index_plus_ws = index + i;
                break;
            }
        }

        if trim_end {
            SnippetState::LineEnd(input[0..=index_minus_ws].concat(), index_plus_ws + 1)
        } else {
            SnippetState::LineEnd(input[0..=index_plus_ws].concat(), index_plus_ws + 1)
        }
    };

    // find a first index where the unicode width of input[0..x] become > max_width
    let max_width_index_in_input = {
        let mut cur_width = 0;
        let mut cur_index = 0;
        for (i, grapheme) in input.iter().enumerate() {
            cur_width += unicode_str_width(grapheme);
            cur_index = i;
            if cur_width > max_width {
                break;
            }
        }
        cur_index
    };
    if max_width_index_in_input == 0 {
        return SnippetState::EndOfInput(input.concat());
    }

    // Find the position in input for breaking the string
    if line_end.is_empty()
        && trim_end
        && !is_whitespace(input[max_width_index_in_input - 1])
        && is_whitespace(input[max_width_index_in_input])
    {
        // At a breaking point already
        // The line won't invalidate the rewriting because:
        // - no extra space needed for the line_end character
        // - extra whitespaces to the right can be trimmed
        return break_at(max_width_index_in_input - 1);
    }
    if let Some(url_index_end) = detect_url(input, max_width_index_in_input) {
        let index_plus_ws = url_index_end
            + input[url_index_end..]
                .iter()
                .skip(1)
                .position(|grapheme| not_whitespace_except_line_feed(grapheme))
                .unwrap_or(0);
        return if trim_end {
            SnippetState::LineEnd(input[..=url_index_end].concat(), index_plus_ws + 1)
        } else {
            SnippetState::LineEnd(input[..=index_plus_ws].concat(), index_plus_ws + 1)
        };
    }

    match input[0..max_width_index_in_input]
        .iter()
        .rposition(|grapheme| is_whitespace(grapheme))
    {
        // Found a whitespace and what is on its left side is big enough.
        Some(index) if index >= MIN_STRING => break_at(index),
        // No whitespace found, try looking for a punctuation instead
        _ => match (0..max_width_index_in_input)
            .rev()
            .skip_while(|pos| !is_valid_linebreak(input, *pos))
            .next()
        {
            // Found a punctuation and what is on its left side is big enough.
            Some(index) if index >= MIN_STRING => break_at(index),
            // Either no boundary character was found to the left of `input[max_chars]`, or the line
            // got too small. We try searching for a boundary character to the right.
            _ => match (max_width_index_in_input..input.len())
                .skip_while(|pos| !is_valid_linebreak(input, *pos))
                .next()
            {
                // A boundary was found after the line limit
                Some(index) => break_at(index),
                // No boundary to the right, the input cannot be broken
                None => SnippetState::EndOfInput(input.concat()),
            },
        },
    }
}

fn is_valid_linebreak(input: &[&str], pos: usize) -> bool {
    let is_whitespace = is_whitespace(input[pos]);
    if is_whitespace {
        return true;
    }
    let is_punctuation = is_punctuation(input[pos]);
    if is_punctuation && !is_part_of_type(input, pos) {
        return true;
    }
    false
}

fn is_part_of_type(input: &[&str], pos: usize) -> bool {
    input.get(pos..=pos + 1) == Some(&[":", ":"])
        || input.get(pos.saturating_sub(1)..=pos) == Some(&[":", ":"])
}

fn is_new_line(grapheme: &str) -> bool {
    let bytes = grapheme.as_bytes();
    bytes.starts_with(b"\n") || bytes.starts_with(b"\r\n")
}

fn is_whitespace(grapheme: &str) -> bool {
    grapheme.chars().all(char::is_whitespace)
}

fn is_punctuation(grapheme: &str) -> bool {
    grapheme
        .chars()
        .all(|c| c.general_category() == GeneralCategory::OtherPunctuation)
}

fn graphemes_width(graphemes: &[&str]) -> usize {
    graphemes.iter().map(|s| unicode_str_width(s)).sum()
}

#[cfg(test)]
mod test {
    use super::{SnippetState, StringFormat, break_string, detect_url, rewrite_string};
    use crate::config::Config;
    use crate::shape::{Indent, Shape};
    use unicode_segmentation::UnicodeSegmentation;

    #[test]
    fn issue343() {
        let config = Default::default();
        let fmt = StringFormat::new(Shape::legacy(2, Indent::empty()), &config);
        rewrite_string("eq_", &fmt, 2);
    }

    #[test]
    fn line_break_at_valid_points_test() {
        let string = "[TheName](Dont::break::my::type::That::would::be::very::nice) break here";
        let graphemes = UnicodeSegmentation::graphemes(&*string, false).collect::<Vec<&str>>();
        assert_eq!(
            break_string(20, false, "", &graphemes[..]),
            SnippetState::LineEnd(
                "[TheName](Dont::break::my::type::That::would::be::very::nice) ".to_string(),
                62
            )
        );
    }

    #[test]
    fn should_break_on_whitespace() {
        let string = "Placerat felis. Mauris porta ante sagittis purus.";
        let graphemes = UnicodeSegmentation::graphemes(&*string, false).collect::<Vec<&str>>();
        assert_eq!(
            break_string(20, false, "", &graphemes[..]),
            SnippetState::LineEnd("Placerat felis. ".to_string(), 16)
        );
        assert_eq!(
            break_string(20, true, "", &graphemes[..]),
            SnippetState::LineEnd("Placerat felis.".to_string(), 16)
        );
    }

    #[test]
    fn should_break_on_punctuation() {
        let string = "Placerat_felis._Mauris_porta_ante_sagittis_purus.";
        let graphemes = UnicodeSegmentation::graphemes(&*string, false).collect::<Vec<&str>>();
        assert_eq!(
            break_string(20, false, "", &graphemes[..]),
            SnippetState::LineEnd("Placerat_felis.".to_string(), 15)
        );
    }

    #[test]
    fn should_break_forward() {
        let string = "Venenatis_tellus_vel_tellus. Aliquam aliquam dolor at justo.";
        let graphemes = UnicodeSegmentation::graphemes(&*string, false).collect::<Vec<&str>>();
        assert_eq!(
            break_string(20, false, "", &graphemes[..]),
            SnippetState::LineEnd("Venenatis_tellus_vel_tellus. ".to_string(), 29)
        );
        assert_eq!(
            break_string(20, true, "", &graphemes[..]),
            SnippetState::LineEnd("Venenatis_tellus_vel_tellus.".to_string(), 29)
        );
    }

    #[test]
    fn nothing_to_break() {
        let string = "Venenatis_tellus_vel_tellus";
        let graphemes = UnicodeSegmentation::graphemes(&*string, false).collect::<Vec<&str>>();
        assert_eq!(
            break_string(20, false, "", &graphemes[..]),
            SnippetState::EndOfInput("Venenatis_tellus_vel_tellus".to_string())
        );
    }

    #[test]
    fn significant_whitespaces() {
        let string = "Neque in sem.      \n      Pellentesque tellus augue.";
        let graphemes = UnicodeSegmentation::graphemes(&*string, false).collect::<Vec<&str>>();
        assert_eq!(
            break_string(15, false, "", &graphemes[..]),
            SnippetState::EndWithLineFeed("Neque in sem.      \n".to_string(), 20)
        );
        assert_eq!(
            break_string(25, false, "", &graphemes[..]),
            SnippetState::EndWithLineFeed("Neque in sem.      \n".to_string(), 20)
        );

        assert_eq!(
            break_string(15, true, "", &graphemes[..]),
            SnippetState::LineEnd("Neque in sem.".to_string(), 19)
        );
        assert_eq!(
            break_string(25, true, "", &graphemes[..]),
            SnippetState::EndWithLineFeed("Neque in sem.\n".to_string(), 20)
        );
    }

    #[test]
    fn big_whitespace() {
        let string = "Neque in sem.            Pellentesque tellus augue.";
        let graphemes = UnicodeSegmentation::graphemes(&*string, false).collect::<Vec<&str>>();
        assert_eq!(
            break_string(20, false, "", &graphemes[..]),
            SnippetState::LineEnd("Neque in sem.            ".to_string(), 25)
        );
        assert_eq!(
            break_string(20, true, "", &graphemes[..]),
            SnippetState::LineEnd("Neque in sem.".to_string(), 25)
        );
    }

    #[test]
    fn newline_in_candidate_line() {
        let string = "Nulla\nconsequat erat at massa. Vivamus id mi.";

        let graphemes = UnicodeSegmentation::graphemes(&*string, false).collect::<Vec<&str>>();
        assert_eq!(
            break_string(25, false, "", &graphemes[..]),
            SnippetState::EndWithLineFeed("Nulla\n".to_string(), 6)
        );
        assert_eq!(
            break_string(25, true, "", &graphemes[..]),
            SnippetState::EndWithLineFeed("Nulla\n".to_string(), 6)
        );

        let mut config: Config = Default::default();
        config.set().max_width(27);
        let fmt = StringFormat::new(Shape::legacy(25, Indent::empty()), &config);
        let rewritten_string = rewrite_string(string, &fmt, 27);
        assert_eq!(
            rewritten_string,
            Some("\"Nulla\nconsequat erat at massa. \\\n Vivamus id mi.\"".to_string())
        );
    }

    #[test]
    fn last_line_fit_with_trailing_whitespaces() {
        let string = "Vivamus id mi.  ";
        let config: Config = Default::default();
        let mut fmt = StringFormat::new(Shape::legacy(25, Indent::empty()), &config);

        fmt.trim_end = true;
        let rewritten_string = rewrite_string(string, &fmt, 25);
        assert_eq!(rewritten_string, Some("\"Vivamus id mi.\"".to_string()));

        fmt.trim_end = false; // default value of trim_end
        let rewritten_string = rewrite_string(string, &fmt, 25);
        assert_eq!(rewritten_string, Some("\"Vivamus id mi.  \"".to_string()));
    }

    #[test]
    fn last_line_fit_with_newline() {
        let string = "Vivamus id mi.\nVivamus id mi.";
        let config: Config = Default::default();
        let fmt = StringFormat {
            opener: "",
            closer: "",
            line_start: "// ",
            line_end: "",
            shape: Shape::legacy(100, Indent::from_width(&config, 4)),
            trim_end: true,
            config: &config,
        };

        let rewritten_string = rewrite_string(string, &fmt, 100);
        assert_eq!(
            rewritten_string,
            Some("Vivamus id mi.\n    // Vivamus id mi.".to_string())
        );
    }

    #[test]
    fn overflow_in_non_string_content() {
        let comment = "Aenean metus.\nVestibulum ac lacus. Vivamus porttitor";
        let config: Config = Default::default();
        let fmt = StringFormat {
            opener: "",
            closer: "",
            line_start: "// ",
            line_end: "",
            shape: Shape::legacy(30, Indent::from_width(&config, 8)),
            trim_end: true,
            config: &config,
        };

        assert_eq!(
            rewrite_string(comment, &fmt, 30),
            Some(
                "Aenean metus.\n        // Vestibulum ac lacus. Vivamus\n        // porttitor"
                    .to_string()
            )
        );
    }

    #[test]
    fn overflow_in_non_string_content_with_line_end() {
        let comment = "Aenean metus.\nVestibulum ac lacus. Vivamus porttitor";
        let config: Config = Default::default();
        let fmt = StringFormat {
            opener: "",
            closer: "",
            line_start: "// ",
            line_end: "@",
            shape: Shape::legacy(30, Indent::from_width(&config, 8)),
            trim_end: true,
            config: &config,
        };

        assert_eq!(
            rewrite_string(comment, &fmt, 30),
            Some(
                "Aenean metus.\n        // Vestibulum ac lacus. Vivamus@\n        // porttitor"
                    .to_string()
            )
        );
    }

    #[test]
    fn blank_line_with_non_empty_line_start() {
        let config: Config = Default::default();
        let mut fmt = StringFormat {
            opener: "",
            closer: "",
            line_start: "// ",
            line_end: "",
            shape: Shape::legacy(30, Indent::from_width(&config, 4)),
            trim_end: true,
            config: &config,
        };

        let comment = "Aenean metus. Vestibulum\n\nac lacus. Vivamus porttitor";
        assert_eq!(
            rewrite_string(comment, &fmt, 30),
            Some(
                "Aenean metus. Vestibulum\n    //\n    // ac lacus. Vivamus porttitor".to_string()
            )
        );

        fmt.shape = Shape::legacy(15, Indent::from_width(&config, 4));
        let comment = "Aenean\n\nmetus. Vestibulum ac lacus. Vivamus porttitor";
        assert_eq!(
            rewrite_string(comment, &fmt, 15),
            Some(
                r#"Aenean
    //
    // metus. Vestibulum
    // ac lacus. Vivamus
    // porttitor"#
                    .to_string()
            )
        );
    }

    #[test]
    fn retain_blank_lines() {
        let config: Config = Default::default();
        let fmt = StringFormat {
            opener: "",
            closer: "",
            line_start: "// ",
            line_end: "",
            shape: Shape::legacy(20, Indent::from_width(&config, 4)),
            trim_end: true,
            config: &config,
        };

        let comment = "Aenean\n\nmetus. Vestibulum ac lacus.\n\n";
        assert_eq!(
            rewrite_string(comment, &fmt, 20),
            Some(
                "Aenean\n    //\n    // metus. Vestibulum ac\n    // lacus.\n    //\n".to_string()
            )
        );

        let comment = "Aenean\n\nmetus. Vestibulum ac lacus.\n";
        assert_eq!(
            rewrite_string(comment, &fmt, 20),
            Some("Aenean\n    //\n    // metus. Vestibulum ac\n    // lacus.\n".to_string())
        );

        let comment = "Aenean\n        \nmetus. Vestibulum ac lacus.";
        assert_eq!(
            rewrite_string(comment, &fmt, 20),
            Some("Aenean\n    //\n    // metus. Vestibulum ac\n    // lacus.".to_string())
        );
    }

    #[test]
    fn boundary_on_edge() {
        let config: Config = Default::default();
        let mut fmt = StringFormat {
            opener: "",
            closer: "",
            line_start: "// ",
            line_end: "",
            shape: Shape::legacy(13, Indent::from_width(&config, 4)),
            trim_end: true,
            config: &config,
        };

        let comment = "Aenean metus. Vestibulum ac lacus.";
        assert_eq!(
            rewrite_string(comment, &fmt, 13),
            Some("Aenean metus.\n    // Vestibulum ac\n    // lacus.".to_string())
        );

        fmt.trim_end = false;
        let comment = "Vestibulum ac lacus.";
        assert_eq!(
            rewrite_string(comment, &fmt, 13),
            Some("Vestibulum \n    // ac lacus.".to_string())
        );

        fmt.trim_end = true;
        fmt.line_end = "\\";
        let comment = "Vestibulum ac lacus.";
        assert_eq!(
            rewrite_string(comment, &fmt, 13),
            Some("Vestibulum\\\n    // ac lacus.".to_string())
        );
    }

    #[test]
    fn detect_urls() {
        let string = "aaa http://example.org something";
        let graphemes = UnicodeSegmentation::graphemes(&*string, false).collect::<Vec<&str>>();
        assert_eq!(detect_url(&graphemes, 8), Some(21));

        let string = "https://example.org something";
        let graphemes = UnicodeSegmentation::graphemes(&*string, false).collect::<Vec<&str>>();
        assert_eq!(detect_url(&graphemes, 0), Some(18));

        let string = "aaa ftp://example.org something";
        let graphemes = UnicodeSegmentation::graphemes(&*string, false).collect::<Vec<&str>>();
        assert_eq!(detect_url(&graphemes, 8), Some(20));

        let string = "aaa file://example.org something";
        let graphemes = UnicodeSegmentation::graphemes(&*string, false).collect::<Vec<&str>>();
        assert_eq!(detect_url(&graphemes, 8), Some(21));

        let string = "aaa http not an url";
        let graphemes = UnicodeSegmentation::graphemes(&*string, false).collect::<Vec<&str>>();
        assert_eq!(detect_url(&graphemes, 6), None);

        let string = "aaa file://example.org";
        let graphemes = UnicodeSegmentation::graphemes(&*string, false).collect::<Vec<&str>>();
        assert_eq!(detect_url(&graphemes, 8), Some(21));
    }
}
