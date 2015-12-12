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

use Indent;
use config::Config;
use utils::round_up_to_power_of_two;

use MIN_STRING;

pub struct StringFormat<'a> {
    pub opener: &'a str,
    pub closer: &'a str,
    pub line_start: &'a str,
    pub line_end: &'a str,
    pub width: usize,
    pub offset: Indent,
    pub trim_end: bool,
    pub config: &'a Config,
}

// FIXME: simplify this!
pub fn rewrite_string<'a>(s: &str, fmt: &StringFormat<'a>) -> Option<String> {
    // Strip line breaks.
    let re = Regex::new(r"(\\[\n\r][:space:]*)").unwrap();
    let stripped_str = re.replace_all(s, "");

    let graphemes = UnicodeSegmentation::graphemes(&*stripped_str, false).collect::<Vec<&str>>();
    let indent = fmt.offset.to_string(fmt.config);
    let punctuation = ":,;.";

    let mut cur_start = 0;
    let mut result = String::with_capacity(round_up_to_power_of_two(s.len()));
    result.push_str(fmt.opener);

    let ender_length = fmt.line_end.len();
    // If we cannot put at least a single character per line, the rewrite won't
    // succeed.
    let max_chars = try_opt!(fmt.width.checked_sub(fmt.opener.len() + ender_length + 1)) + 1;

    loop {
        let mut cur_end = cur_start + max_chars;

        if cur_end >= graphemes.len() {
            let line = &graphemes[cur_start..].join("");
            result.push_str(line);
            break;
        }

        // Push cur_end left until we reach whitespace.
        while !graphemes[cur_end - 1].trim().is_empty() {
            cur_end -= 1;
            if cur_end - cur_start < MIN_STRING {
                cur_end = cur_start + max_chars;
                // Look for punctuation to break on.
                while (!punctuation.contains(graphemes[cur_end - 1])) && cur_end > 1 {
                    cur_end -= 1;
                }
                // We can't break at whitespace or punctuation, fall back to splitting
                // anywhere that doesn't break an escape sequence.
                if cur_end < cur_start + MIN_STRING {
                    cur_end = cur_start + max_chars;
                    while graphemes[cur_end - 1] == "\\" && cur_end > 1 {
                        cur_end -= 1;
                    }
                }
                break;
            }
        }
        // Make sure there is no whitespace to the right of the break.
        while cur_end < s.len() && graphemes[cur_end].trim().is_empty() {
            cur_end += 1;
        }
        let raw_line = graphemes[cur_start..cur_end].join("");
        let line = if fmt.trim_end {
            raw_line.trim()
        } else {
            // FIXME: use as_str once it's stable.
            &*raw_line
        };

        result.push_str(line);
        result.push_str(fmt.line_end);
        result.push('\n');
        result.push_str(&indent);
        result.push_str(fmt.line_start);

        cur_start = cur_end;
    }
    result.push_str(fmt.closer);

    Some(result)
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
            width: 2,
            offset: ::Indent::empty(),
            trim_end: false,
            config: &config,
        };

        rewrite_string("eq_", &fmt);
    }
}
