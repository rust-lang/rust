// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Format comments.

use string::{StringFormat, rewrite_string};
use utils::make_indent;

pub fn rewrite_comment(orig: &str, block_style: bool, width: usize, offset: usize) -> String {
    let s = orig.trim();

    // Edge case: block comments. Let's not trim their lines (for now).
    let opener = if block_style { "/* " } else { "// " };
    let closer = if block_style { " */" } else { "" };
    let line_start = if block_style { " * " } else { "// " };

    let max_chars = width.checked_sub(closer.len()).unwrap_or(1)
                         .checked_sub(opener.len()).unwrap_or(1);

    let fmt = StringFormat {
        opener: "",
        closer: "",
        line_start: line_start,
        line_end: "",
        width: max_chars,
        offset: offset + opener.len() - line_start.len(),
        trim_end: true
    };

    let indent_str = make_indent(offset);
    let line_breaks = s.chars().filter(|&c| c == '\n').count();

    let (_, mut s) = s.lines().enumerate()
        .map(|(i, mut line)| {
            line = line.trim();

            // Drop old closer.
            if i == line_breaks && line.ends_with("*/") && !line.starts_with("//") {
                line = &line[..(line.len() - 2)];
            }

            line.trim_right_matches(' ')
        })
        .map(left_trim_comment_line)
        .fold((true, opener.to_owned()), |(first, mut acc), line| {
            if !first {
                acc.push('\n');
                acc.push_str(&indent_str);
                acc.push_str(line_start);
            }

            if line.len() > max_chars {
                acc.push_str(&rewrite_string(line, &fmt));
            } else {
                acc.push_str(line);
            }

            (false, acc)
        });

    s.push_str(closer);

    s
}

fn left_trim_comment_line<'a>(line: &'a str) -> &'a str {
    if line.starts_with("/* ") || line.starts_with("// ") {
        &line[3..]
    } else if line.starts_with("/*") || line.starts_with("* ") || line.starts_with("//") {
        &line[2..]
    } else if line.starts_with("*") {
        &line[1..]
    } else {
        line
    }
}

#[test]
fn format_comments() {
    assert_eq!("/* test */", rewrite_comment(" //test", true, 100, 100));
    assert_eq!("// comment\n// on a", rewrite_comment("// comment on a", false, 10, 0));

    assert_eq!("//  A multi line comment\n            // between args.",
               rewrite_comment("//  A multi line comment\n             // between args.",
                               false,
                               60,
                               12));

    let input = "// comment";
    let expected_output = "/* com\n                                                                      \
                            * men\n                                                                      \
                            * t */";
    assert_eq!(expected_output, rewrite_comment(input, true, 9, 69));
}


pub trait FindUncommented {
    fn find_uncommented(&self, pat: &str) -> Option<usize>;
}

impl FindUncommented for str {
    fn find_uncommented(&self, pat: &str) -> Option<usize> {
        let mut needle_iter = pat.chars();
        let mut possible_comment = false;

        for (i, b) in self.char_indices() {
            match needle_iter.next() {
                Some(c) => {
                    if b != c {
                        needle_iter = pat.chars();
                    }
                },
                None => return Some(i - pat.len())
            }

            if possible_comment && (b == '/' || b == '*') {
                return find_comment_end(&self[(i-1)..])
                    .and_then(|end| {
                        self[(end + i - 1)..].find_uncommented(pat)
                                             .map(|idx| idx + end + i - 1)
                    });
            }

            possible_comment = b == '/';
        }

        // Handle case where the pattern is a suffix of the search string
        match needle_iter.next() {
            Some(_) => None,
            None => Some(self.len() - pat.len())
        }
    }
}

#[test]
fn test_find_uncommented() {
    fn check(haystack: &str, needle: &str, expected: Option<usize>) {
        println!("haystack {:?}, needle: {:?}", haystack, needle);
        assert_eq!(expected, haystack.find_uncommented(needle));
    }

    check("/*/ */test", "test", Some(6));
    check("//test\ntest", "test", Some(7));
    check("/* comment only */", "whatever", None);
    check("/* comment */ some text /* more commentary */ result", "result", Some(46));
    check("sup // sup", "p", Some(2));
    check("sup", "x", None);
    check("π? /**/ π is nice!", "π is nice", Some(9));
    check("/*sup yo? \n sup*/ sup", "p", Some(20));
    check("hel/*lohello*/lo", "hello", None);
    check("acb", "ab", None);
}

// Returns the first byte position after the first comment. The given string
// is expected to be prefixed by a comment, including delimiters.
// Good: "/* /* inner */ outer */ code();"
// Bad:  "code(); // hello\n world!"
pub fn find_comment_end(s: &str) -> Option<usize> {
    if s.starts_with("//") {
        s.find('\n').map(|idx| idx + 1)
    } else {
        // Block comment
        let mut levels = 0;
        let mut prev_char = 'a';

        for (i, mut c) in s.char_indices() {
            if c == '*' && prev_char == '/' {
                levels += 1;
                c = 'a'; // Invalidate prev_char
            } else if c == '/' && prev_char == '*' {
                levels -= 1;

                if levels == 0 {
                    return Some(i + 1);
                }
                c = 'a';
            }

            prev_char = c;
        }

        None
    }
}

#[test]
fn comment_end() {
    assert_eq!(Some(6), find_comment_end("// hi\n"));
    assert_eq!(Some(9), find_comment_end("/* sup */ "));
    assert_eq!(Some(9), find_comment_end("/*/**/ */ "));
    assert_eq!(Some(6), find_comment_end("/*/ */ weird!"));
    assert_eq!(None, find_comment_end("/* hi /* test */"));
    assert_eq!(None, find_comment_end("// hi /* test */"));
    assert_eq!(Some(9), find_comment_end("// hi /*\n."));
}
