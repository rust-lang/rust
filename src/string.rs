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

use unicode_segmentation::UnicodeSegmentation;
use regex::Regex;

use Shape;
use config::Config;
use utils::wrap_str;

use MIN_STRING;

pub struct StringFormat<'a> {
    pub opener: &'a str,
    pub closer: &'a str,
    pub line_start: &'a str,
    pub line_end: &'a str,
    pub shape: Shape,
    pub trim_end: bool,
    pub config: &'a Config,
}

// FIXME: simplify this!
pub fn rewrite_string<'a>(orig: &str, fmt: &StringFormat<'a>) -> Option<String> {
    // Strip line breaks.
    let re = Regex::new(r"([^\\](\\\\)*)\\[\n\r][[:space:]]*").unwrap();
    let stripped_str = re.replace_all(orig, "$1");

    let graphemes = UnicodeSegmentation::graphemes(&*stripped_str, false).collect::<Vec<&str>>();
    let shape = fmt.shape.visual_indent(0);
    let indent = shape.indent.to_string(fmt.config);
    let punctuation = ":,;.";

    // `cur_start` is the position in `orig` of the start of the current line.
    let mut cur_start = 0;
    let mut result =
        String::with_capacity(stripped_str.len().checked_next_power_of_two().unwrap_or(
            usize::max_value(),
        ));
    result.push_str(fmt.opener);

    let ender_length = fmt.line_end.len();
    // If we cannot put at least a single character per line, the rewrite won't
    // succeed.
    let max_chars = try_opt!(shape.width.checked_sub(fmt.opener.len() + ender_length + 1)) + 1;

    // Snip a line at a time from `orig` until it is used up. Push the snippet
    // onto result.
    'outer: loop {
        // `cur_end` will be where we break the line, as an offset into `orig`.
        // Initialised to the maximum it could be (which may be beyond `orig`).
        let mut cur_end = cur_start + max_chars;

        // We can fit the rest of the string on this line, so we're done.
        if cur_end >= graphemes.len() {
            let line = &graphemes[cur_start..].join("");
            result.push_str(line);
            break 'outer;
        }

        // Push cur_end left until we reach whitespace (or the line is too small).
        while !graphemes[cur_end - 1].trim().is_empty() {
            cur_end -= 1;
            if cur_end < cur_start + MIN_STRING {
                // We couldn't find whitespace before the string got too small.
                // So start again at the max length and look for punctuation.
                cur_end = cur_start + max_chars;
                while !punctuation.contains(graphemes[cur_end - 1]) {
                    cur_end -= 1;

                    // If we can't break at whitespace or punctuation, grow the string instead.
                    if cur_end < cur_start + MIN_STRING {
                        cur_end = cur_start + max_chars;
                        while !(punctuation.contains(graphemes[cur_end - 1]) ||
                                    graphemes[cur_end - 1].trim().is_empty())
                        {
                            if cur_end >= graphemes.len() {
                                let line = &graphemes[cur_start..].join("");
                                result.push_str(line);
                                break 'outer;
                            }
                            cur_end += 1;
                        }
                        break;
                    }
                }
                break;
            }
        }
        // Make sure there is no whitespace to the right of the break.
        while cur_end < stripped_str.len() && graphemes[cur_end].trim().is_empty() {
            cur_end += 1;
        }

        // Make the current line and add it on to result.
        let raw_line = graphemes[cur_start..cur_end].join("");
        let line = if fmt.trim_end {
            raw_line.trim()
        } else {
            raw_line.as_str()
        };

        result.push_str(line);
        result.push_str(fmt.line_end);
        result.push('\n');
        result.push_str(&indent);
        result.push_str(fmt.line_start);

        // The next line starts where the current line ends.
        cur_start = cur_end;
    }

    result.push_str(fmt.closer);
    wrap_str(result, fmt.config.max_width(), fmt.shape)
}

#[cfg(test)]
mod test {
    use super::{StringFormat, rewrite_string};

    #[test]
    fn issue343() {
        let config = Default::default();
        let fmt = StringFormat {
            opener: "\"",
            closer: "\"",
            line_start: " ",
            line_end: "\\",
            shape: ::Shape::legacy(2, ::Indent::empty()),
            trim_end: false,
            config: &config,
        };

        rewrite_string("eq_", &fmt);
    }
}
