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

use std::iter;

use Indent;
use config::Config;
use string::{StringFormat, rewrite_string};

pub fn rewrite_comment(orig: &str,
                       block_style: bool,
                       width: usize,
                       offset: Indent,
                       config: &Config)
                       -> Option<String> {
    let s = orig.trim();

    // Edge case: block comments. Let's not trim their lines (for now).
    let (opener, closer, line_start) = if block_style {
        ("/* ", " */", " * ")
    } else if orig.starts_with("///") {
        ("/// ", "", "/// ")
    } else if orig.starts_with("//!") {
        ("//! ", "", "//! ")
    } else {
        ("// ", "", "// ")
    };

    let max_chars = width.checked_sub(closer.len() + opener.len()).unwrap_or(1);

    let fmt = StringFormat {
        opener: "",
        closer: "",
        line_start: line_start,
        line_end: "",
        width: max_chars,
        offset: offset + (opener.len() - line_start.len()),
        trim_end: true,
        config: config,
    };

    let indent_str = offset.to_string(config);
    let line_breaks = s.chars().filter(|&c| c == '\n').count();

    let lines = s.lines()
                 .enumerate()
                 .map(|(i, mut line)| {
                     line = line.trim();
                     // Drop old closer.
                     if i == line_breaks && line.ends_with("*/") && !line.starts_with("//") {
                         line = &line[..(line.len() - 2)];
                     }

                     line.trim_right()
                 })
                 .map(left_trim_comment_line)
                 .map(|line| {
                     if line_breaks == 0 {
                         line.trim_left()
                     } else {
                         line
                     }
                 });

    let mut result = opener.to_owned();
    let mut first = true;

    for line in lines {
        if !first {
            result.push('\n');
            result.push_str(&indent_str);
            result.push_str(line_start);
        }

        if config.wrap_comments && line.len() > max_chars {
            let rewrite = try_opt!(rewrite_string(line, &fmt));
            result.push_str(&rewrite);
        } else {
            if line.len() == 0 {
                // Remove space if this is an empty comment or a doc comment.
                result.pop();
            }
            result.push_str(line);
        }

        first = false;
    }

    result.push_str(closer);

    Some(result)
}

fn left_trim_comment_line(line: &str) -> &str {
    if line.starts_with("//! ") || line.starts_with("/// ") {
        &line[4..]
    } else if line.starts_with("/* ") || line.starts_with("// ") || line.starts_with("//!") ||
       line.starts_with("///") {
        &line[3..]
    } else if line.starts_with("/*") || line.starts_with("* ") || line.starts_with("//") {
        &line[2..]
    } else if line.starts_with("*") {
        &line[1..]
    } else {
        line
    }
}

pub trait FindUncommented {
    fn find_uncommented(&self, pat: &str) -> Option<usize>;
}

impl FindUncommented for str {
    fn find_uncommented(&self, pat: &str) -> Option<usize> {
        let mut needle_iter = pat.chars();
        for (kind, (i, b)) in CharClasses::new(self.char_indices()) {
            match needle_iter.next() {
                None => {
                    return Some(i - pat.len());
                }
                Some(c) => {
                    match kind {
                        CodeCharKind::Normal if b == c => {}
                        _ => {
                            needle_iter = pat.chars();
                        }
                    }
                }
            }
        }

        // Handle case where the pattern is a suffix of the search string
        match needle_iter.next() {
            Some(_) => None,
            None => Some(self.len() - pat.len()),
        }
    }
}

// Returns the first byte position after the first comment. The given string
// is expected to be prefixed by a comment, including delimiters.
// Good: "/* /* inner */ outer */ code();"
// Bad:  "code(); // hello\n world!"
pub fn find_comment_end(s: &str) -> Option<usize> {
    let mut iter = CharClasses::new(s.char_indices());
    for (kind, (i, _c)) in &mut iter {
        if kind == CodeCharKind::Normal {
            return Some(i);
        }
    }

    // Handle case where the comment ends at the end of s.
    if iter.status == CharClassesStatus::Normal {
        Some(s.len())
    } else {
        None
    }
}

/// Returns true if text contains any comment.
pub fn contains_comment(text: &str) -> bool {
    CharClasses::new(text.chars()).any(|(kind, _)| kind == CodeCharKind::Comment)
}

struct CharClasses<T>
    where T: Iterator,
          T::Item: RichChar
{
    base: iter::Peekable<T>,
    status: CharClassesStatus,
}

trait RichChar {
    fn get_char(&self) -> char;
}

impl RichChar for char {
    fn get_char(&self) -> char {
        *self
    }
}

impl RichChar for (usize, char) {
    fn get_char(&self) -> char {
        self.1
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
enum CharClassesStatus {
    Normal,
    LitString,
    LitStringEscape,
    LitChar,
    LitCharEscape,
    // The u32 is the nesting deepness of the comment
    BlockComment(u32),
    // Status when the '/' has been consumed, but not yet the '*', deepness is
    // the new deepness (after the comment opening).
    BlockCommentOpening(u32),
    // Status when the '*' has been consumed, but not yet the '/', deepness is
    // the new deepness (after the comment closing).
    BlockCommentClosing(u32),
    LineComment,
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum CodeCharKind {
    Normal,
    Comment,
}

impl<T> CharClasses<T>
    where T: Iterator,
          T::Item: RichChar
{
    fn new(base: T) -> CharClasses<T> {
        CharClasses {
            base: base.peekable(),
            status: CharClassesStatus::Normal,
        }
    }
}

impl<T> Iterator for CharClasses<T>
    where T: Iterator,
          T::Item: RichChar
{
    type Item = (CodeCharKind, T::Item);

    fn next(&mut self) -> Option<(CodeCharKind, T::Item)> {
        let item = try_opt!(self.base.next());
        let chr = item.get_char();
        self.status = match self.status {
            CharClassesStatus::LitString => {
                match chr {
                    '"' => CharClassesStatus::Normal,
                    '\\' => CharClassesStatus::LitStringEscape,
                    _ => CharClassesStatus::LitString,
                }
            }
            CharClassesStatus::LitStringEscape => CharClassesStatus::LitString,
            CharClassesStatus::LitChar => {
                match chr {
                    '\\' => CharClassesStatus::LitCharEscape,
                    '\'' => CharClassesStatus::Normal,
                    _ => CharClassesStatus::LitChar,
                }
            }
            CharClassesStatus::LitCharEscape => CharClassesStatus::LitChar,
            CharClassesStatus::Normal => {
                match chr {
                    '"' => CharClassesStatus::LitString,
                    '\'' => CharClassesStatus::LitChar,
                    '/' => {
                        match self.base.peek() {
                            Some(next) if next.get_char() == '*' => {
                                self.status = CharClassesStatus::BlockCommentOpening(1);
                                return Some((CodeCharKind::Comment, item));
                            }
                            Some(next) if next.get_char() == '/' => {
                                self.status = CharClassesStatus::LineComment;
                                return Some((CodeCharKind::Comment, item));
                            }
                            _ => CharClassesStatus::Normal,
                        }
                    }
                    _ => CharClassesStatus::Normal,
                }
            }
            CharClassesStatus::BlockComment(deepness) => {
                if deepness == 0 {
                    // This is the closing '/'
                    assert_eq!(chr, '/');
                    self.status = CharClassesStatus::Normal;
                    return Some((CodeCharKind::Comment, item));
                }
                self.status = match self.base.peek() {
                    Some(next) if next.get_char() == '/' && chr == '*' => {
                        CharClassesStatus::BlockCommentClosing(deepness - 1)
                    }
                    Some(next) if next.get_char() == '*' && chr == '/' => {
                        CharClassesStatus::BlockCommentOpening(deepness + 1)
                    }
                    _ => CharClassesStatus::BlockComment(deepness),
                };
                return Some((CodeCharKind::Comment, item));
            }
            CharClassesStatus::BlockCommentOpening(deepness) => {
                assert_eq!(chr, '*');
                self.status = CharClassesStatus::BlockComment(deepness);
                return Some((CodeCharKind::Comment, item));
            }
            CharClassesStatus::BlockCommentClosing(deepness) => {
                assert_eq!(chr, '/');
                self.status = if deepness == 0 {
                    CharClassesStatus::Normal
                } else {
                    CharClassesStatus::BlockComment(deepness)
                };
                return Some((CodeCharKind::Comment, item));
            }
            CharClassesStatus::LineComment => {
                self.status = match chr {
                    '\n' => CharClassesStatus::Normal,
                    _ => CharClassesStatus::LineComment,
                };
                return Some((CodeCharKind::Comment, item));
            }
        };
        Some((CodeCharKind::Normal, item))
    }
}

/// Iterator over an alternating sequence of functional and commented parts of
/// a string. The first item is always a, possibly zero length, subslice of
/// functional text. Line style comments contain their ending newlines.
pub struct CommentCodeSlices<'a> {
    slice: &'a str,
    last_slice_kind: CodeCharKind,
    last_slice_end: usize,
}

impl<'a> CommentCodeSlices<'a> {
    pub fn new(slice: &'a str) -> CommentCodeSlices<'a> {
        CommentCodeSlices {
            slice: slice,
            last_slice_kind: CodeCharKind::Comment,
            last_slice_end: 0,
        }
    }
}

impl<'a> Iterator for CommentCodeSlices<'a> {
    type Item = (CodeCharKind, usize, &'a str);

    fn next(&mut self) -> Option<Self::Item> {
        if self.last_slice_end == self.slice.len() {
            return None;
        }

        let mut sub_slice_end = self.last_slice_end;
        let mut first_whitespace = None;
        let subslice = &self.slice[self.last_slice_end..];
        let mut iter = CharClasses::new(subslice.char_indices());

        for (kind, (i, c)) in &mut iter {
            let is_comment_connector = self.last_slice_kind == CodeCharKind::Normal &&
                                       &subslice[..2] == "//" &&
                                       [' ', '\t'].contains(&c);

            if is_comment_connector && first_whitespace.is_none() {
                first_whitespace = Some(i);
            }

            if kind == self.last_slice_kind && !is_comment_connector {
                let last_index = match first_whitespace {
                    Some(j) => j,
                    None => i,
                };
                sub_slice_end = self.last_slice_end + last_index;
                break;
            }

            if !is_comment_connector {
                first_whitespace = None;
            }
        }

        if let (None, true) = (iter.next(), sub_slice_end == self.last_slice_end) {
            // This was the last subslice.
            sub_slice_end = match first_whitespace {
                Some(i) => self.last_slice_end + i,
                None => self.slice.len(),
            };
        }

        let kind = match self.last_slice_kind {
            CodeCharKind::Comment => CodeCharKind::Normal,
            CodeCharKind::Normal => CodeCharKind::Comment,
        };
        let res = (kind,
                   self.last_slice_end,
                   &self.slice[self.last_slice_end..sub_slice_end]);
        self.last_slice_end = sub_slice_end;
        self.last_slice_kind = kind;

        Some(res)
    }
}

#[cfg(test)]
mod test {
    use super::{CharClasses, CodeCharKind, contains_comment, rewrite_comment, FindUncommented,
                CommentCodeSlices};
    use Indent;

    #[test]
    fn char_classes() {
        let mut iter = CharClasses::new("//\n\n".chars());

        assert_eq!((CodeCharKind::Comment, '/'), iter.next().unwrap());
        assert_eq!((CodeCharKind::Comment, '/'), iter.next().unwrap());
        assert_eq!((CodeCharKind::Comment, '\n'), iter.next().unwrap());
        assert_eq!((CodeCharKind::Normal, '\n'), iter.next().unwrap());
        assert_eq!(None, iter.next());
    }

    #[test]
    fn comment_code_slices() {
        let input = "code(); /* test */ 1 + 1";
        let mut iter = CommentCodeSlices::new(input);

        assert_eq!((CodeCharKind::Normal, 0, "code(); "), iter.next().unwrap());
        assert_eq!((CodeCharKind::Comment, 8, "/* test */"),
                   iter.next().unwrap());
        assert_eq!((CodeCharKind::Normal, 18, " 1 + 1"), iter.next().unwrap());
        assert_eq!(None, iter.next());
    }

    #[test]
    fn comment_code_slices_two() {
        let input = "// comment\n    test();";
        let mut iter = CommentCodeSlices::new(input);

        assert_eq!((CodeCharKind::Normal, 0, ""), iter.next().unwrap());
        assert_eq!((CodeCharKind::Comment, 0, "// comment\n"),
                   iter.next().unwrap());
        assert_eq!((CodeCharKind::Normal, 11, "    test();"),
                   iter.next().unwrap());
        assert_eq!(None, iter.next());
    }

    #[test]
    fn comment_code_slices_three() {
        let input = "1 // comment\n    // comment2\n\n";
        let mut iter = CommentCodeSlices::new(input);

        assert_eq!((CodeCharKind::Normal, 0, "1 "), iter.next().unwrap());
        assert_eq!((CodeCharKind::Comment, 2, "// comment\n    // comment2\n"),
                   iter.next().unwrap());
        assert_eq!((CodeCharKind::Normal, 29, "\n"), iter.next().unwrap());
        assert_eq!(None, iter.next());
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn format_comments() {
        let mut config: ::config::Config = Default::default();
        config.wrap_comments = true;
        assert_eq!("/* test */", rewrite_comment(" //test", true, 100, Indent::new(0, 100),
                                                 &config).unwrap());
        assert_eq!("// comment\n// on a", rewrite_comment("// comment on a", false, 10,
                                                          Indent::empty(), &config).unwrap());

        assert_eq!("//  A multi line comment\n            // between args.",
                   rewrite_comment("//  A multi line comment\n             // between args.",
                                   false,
                                   60,
                                   Indent::new(0, 12),
                                   &config).unwrap());

        let input = "// comment";
        let expected =
            "/* com\n                                                                      \
             * men\n                                                                      \
             * t */";
        assert_eq!(expected, rewrite_comment(input, true, 9, Indent::new(0, 69), &config).unwrap());

        assert_eq!("/* trimmed */", rewrite_comment("/*   trimmed    */", true, 100,
                                                    Indent::new(0, 100), &config).unwrap());
    }

    // This is probably intended to be a non-test fn, but it is not used. I'm
    // keeping it around unless it helps us test stuff.
    fn uncommented(text: &str) -> String {
        CharClasses::new(text.chars())
            .filter_map(|(s, c)| {
                match s {
                    CodeCharKind::Normal => Some(c),
                    CodeCharKind::Comment => None,
                }
            })
            .collect()
    }

    #[test]
    fn test_uncommented() {
        assert_eq!(&uncommented("abc/*...*/"), "abc");
        assert_eq!(&uncommented("// .... /* \n../* /* *** / */ */a/* // */c\n"),
                   "..ac\n");
        assert_eq!(&uncommented("abc \" /* */\" qsdf"), "abc \" /* */\" qsdf");
    }

    #[test]
    fn test_contains_comment() {
        assert_eq!(contains_comment("abc"), false);
        assert_eq!(contains_comment("abc // qsdf"), true);
        assert_eq!(contains_comment("abc /* kqsdf"), true);
        assert_eq!(contains_comment("abc \" /* */\" qsdf"), false);
    }

    #[test]
    fn test_find_uncommented() {
        fn check(haystack: &str, needle: &str, expected: Option<usize>) {
            assert_eq!(expected, haystack.find_uncommented(needle));
        }

        check("/*/ */test", "test", Some(6));
        check("//test\ntest", "test", Some(7));
        check("/* comment only */", "whatever", None);
        check("/* comment */ some text /* more commentary */ result",
              "result",
              Some(46));
        check("sup // sup", "p", Some(2));
        check("sup", "x", None);
        check(r#"π? /**/ π is nice!"#, r#"π is nice"#, Some(9));
        check("/*sup yo? \n sup*/ sup", "p", Some(20));
        check("hel/*lohello*/lo", "hello", None);
        check("acb", "ab", None);
        check(",/*A*/ ", ",", Some(0));
        check("abc", "abc", Some(0));
        check("/* abc */", "abc", None);
        check("/**/abc/* */", "abc", Some(4));
        check("\"/* abc */\"", "abc", Some(4));
        check("\"/* abc", "abc", Some(4));
    }
}
