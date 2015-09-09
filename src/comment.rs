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

use string::{StringFormat, rewrite_string};
use utils::make_indent;

pub fn rewrite_comment(orig: &str, block_style: bool, width: usize, offset: usize) -> String {
    let s = orig.trim();

    // Edge case: block comments. Let's not trim their lines (for now).
    let (opener, closer, line_start) = if block_style {
        ("/* ", " */", " * ")
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
        offset: offset + opener.len() - line_start.len(),
        trim_end: true,
    };

    let indent_str = make_indent(offset);
    let line_breaks = s.chars().filter(|&c| c == '\n').count();

    let (_, mut s) = s.lines()
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
        })
        .fold((true, opener.to_owned()), |(first, mut acc), line| {
            if !first {
                acc.push('\n');
                acc.push_str(&indent_str);
                acc.push_str(line_start);
            }

            if line.len() > max_chars {
                acc.push_str(&rewrite_string(line, &fmt));
            } else {
                if line.len() == 0 {
                    acc.pop(); // Remove space if this is an empty comment.
                } else {
                    acc.push_str(line);
                }
            }

            (false, acc)
        });

    s.push_str(closer);

    s
}

fn left_trim_comment_line(line: &str) -> &str {
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
                Some(c) => match kind {
                    CodeCharKind::Normal if b == c => {}
                    _ => {
                        needle_iter = pat.chars();
                    }
                },
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
    // Status when the '/' has been consumed, but not yet the '*', deepness is the new deepness
    // (after the comment opening).
    BlockCommentOpening(u32),
    // Status when the '*' has been consumed, but not yet the '/', deepness is the new deepness
    // (after the comment closing).
    BlockCommentClosing(u32),
    LineComment,
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
enum CodeCharKind {
    Normal,
    Comment,
}

impl<T> CharClasses<T> where T: Iterator, T::Item: RichChar {
    fn new(base: T) -> CharClasses<T> {
        CharClasses { base: base.peekable(), status: CharClassesStatus::Normal }
    }
}

impl<T> Iterator for CharClasses<T> where T: Iterator, T::Item: RichChar {
    type Item = (CodeCharKind, T::Item);

    fn next(&mut self) -> Option<(CodeCharKind, T::Item)> {
        let item = try_opt!(self.base.next());
        let chr = item.get_char();
        self.status = match self.status {
            CharClassesStatus::LitString => match chr {
                '"' => CharClassesStatus::Normal,
                '\\' => CharClassesStatus::LitStringEscape,
                _ => CharClassesStatus::LitString,
            },
            CharClassesStatus::LitStringEscape => CharClassesStatus::LitString,
            CharClassesStatus::LitChar => match chr {
                '\\' => CharClassesStatus::LitCharEscape,
                '\'' => CharClassesStatus::Normal,
                _ => CharClassesStatus::LitChar,
            },
            CharClassesStatus::LitCharEscape => CharClassesStatus::LitChar,
            CharClassesStatus::Normal => {
                match chr {
                    '"' => CharClassesStatus::LitString,
                    '\'' => CharClassesStatus::LitChar,
                    '/' => match self.base.peek() {
                        Some(next) if next.get_char() == '*' => {
                            self.status = CharClassesStatus::BlockCommentOpening(1);
                            return Some((CodeCharKind::Comment, item));
                        }
                        Some(next) if next.get_char() == '/' => {
                            self.status = CharClassesStatus::LineComment;
                            return Some((CodeCharKind::Comment, item));
                        }
                        _ => CharClassesStatus::Normal,
                    },
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
                    Some(next) if next.get_char() == '/' && chr == '*' =>
                        CharClassesStatus::BlockCommentClosing(deepness - 1),
                    Some(next) if next.get_char() == '*' && chr == '/' =>
                        CharClassesStatus::BlockCommentOpening(deepness + 1),
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

#[cfg(test)]
mod test {
    use super::{CharClasses, CodeCharKind, contains_comment, rewrite_comment, FindUncommented};

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
        let expected = "/* com\n                                                                      \
                        * men\n                                                                      \
                        * t */";
        assert_eq!(expected, rewrite_comment(input, true, 9, 69));

        assert_eq!("/* trimmed */", rewrite_comment("/*   trimmed    */", true, 100, 100));
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
        assert_eq!(&uncommented("// .... /* \n../* /* *** / */ */a/* // */c\n"), "..ac\n");
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
            println!("haystack {:?}, needle: {:?}", haystack, needle);
            assert_eq!(expected, haystack.find_uncommented(needle));
        }

        check("/*/ */test", "test", Some(6));
        check("//test\ntest", "test", Some(7));
        check("/* comment only */", "whatever", None);
        check("/* comment */ some text /* more commentary */ result", "result", Some(46));
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
