// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::CommentStyle::*;

use ast;
use codemap::{BytePos, CharPos, CodeMap, Pos};
use diagnostic;
use parse::lexer::is_block_doc_comment;
use parse::lexer::{StringReader, TokenAndSpan};
use parse::lexer::{is_whitespace, Reader};
use parse::lexer;
use print::pprust;
use str::char_at;

use std::io::Read;
use std::usize;

#[derive(Clone, Copy, PartialEq)]
pub enum CommentStyle {
    /// No code on either side of each line of the comment
    Isolated,
    /// Code exists to the left of the comment
    Trailing,
    /// Code before /* foo */ and after the comment
    Mixed,
    /// Just a manual blank line "\n\n", for layout
    BlankLine,
}

#[derive(Clone)]
pub struct Comment {
    pub style: CommentStyle,
    pub lines: Vec<String>,
    pub pos: BytePos,
}

pub fn is_doc_comment(s: &str) -> bool {
    (s.starts_with("///") && super::is_doc_comment(s)) ||
    s.starts_with("//!") ||
    (s.starts_with("/**") && is_block_doc_comment(s)) ||
    s.starts_with("/*!")
}

pub fn doc_comment_style(comment: &str) -> ast::AttrStyle {
    assert!(is_doc_comment(comment));
    if comment.starts_with("//!") || comment.starts_with("/*!") {
        ast::AttrStyle::Inner
    } else {
        ast::AttrStyle::Outer
    }
}

pub fn strip_doc_comment_decoration(comment: &str) -> String {
    /// remove whitespace-only lines from the start/end of lines
    fn vertical_trim(lines: Vec<String>) -> Vec<String> {
        let mut i = 0;
        let mut j = lines.len();
        // first line of all-stars should be omitted
        if !lines.is_empty() &&
                lines[0].chars().all(|c| c == '*') {
            i += 1;
        }
        while i < j && lines[i].trim().is_empty() {
            i += 1;
        }
        // like the first, a last line of all stars should be omitted
        if j > i && lines[j - 1]
                         .chars()
                         .skip(1)
                         .all(|c| c == '*') {
            j -= 1;
        }
        while j > i && lines[j - 1].trim().is_empty() {
            j -= 1;
        }
        lines[i..j].iter().cloned().collect()
    }

    /// remove a "[ \t]*\*" block from each line, if possible
    fn horizontal_trim(lines: Vec<String> ) -> Vec<String> {
        let mut i = usize::MAX;
        let mut can_trim = true;
        let mut first = true;
        for line in &lines {
            for (j, c) in line.chars().enumerate() {
                if j > i || !"* \t".contains(c) {
                    can_trim = false;
                    break;
                }
                if c == '*' {
                    if first {
                        i = j;
                        first = false;
                    } else if i != j {
                        can_trim = false;
                    }
                    break;
                }
            }
            if i > line.len() {
                can_trim = false;
            }
            if !can_trim {
                break;
            }
        }

        if can_trim {
            lines.iter().map(|line| {
                (&line[i + 1..line.len()]).to_string()
            }).collect()
        } else {
            lines
        }
    }

    // one-line comments lose their prefix
    const ONELINERS: &'static [&'static str] = &["///!", "///", "//!", "//"];
    for prefix in ONELINERS {
        if comment.starts_with(*prefix) {
            return (&comment[prefix.len()..]).to_string();
        }
    }

    if comment.starts_with("/*") {
        let lines = comment[3..comment.len() - 2]
            .lines()
            .map(|s| s.to_string())
            .collect::<Vec<String> >();

        let lines = vertical_trim(lines);
        let lines = horizontal_trim(lines);

        return lines.join("\n");
    }

    panic!("not a doc-comment: {}", comment);
}

fn push_blank_line_comment(rdr: &StringReader, comments: &mut Vec<Comment>) {
    debug!(">>> blank-line comment");
    comments.push(Comment {
        style: BlankLine,
        lines: Vec::new(),
        pos: rdr.last_pos,
    });
}

fn consume_whitespace_counting_blank_lines(rdr: &mut StringReader,
                                           comments: &mut Vec<Comment>) {
    while is_whitespace(rdr.curr) && !rdr.is_eof() {
        if rdr.col == CharPos(0) && rdr.curr_is('\n') {
            push_blank_line_comment(rdr, &mut *comments);
        }
        rdr.bump();
    }
}


fn read_shebang_comment(rdr: &mut StringReader, code_to_the_left: bool,
                        comments: &mut Vec<Comment>) {
    debug!(">>> shebang comment");
    let p = rdr.last_pos;
    debug!("<<< shebang comment");
    comments.push(Comment {
        style: if code_to_the_left { Trailing } else { Isolated },
        lines: vec!(rdr.read_one_line_comment()),
        pos: p
    });
}

fn read_line_comments(rdr: &mut StringReader, code_to_the_left: bool,
                      comments: &mut Vec<Comment>) {
    debug!(">>> line comments");
    let p = rdr.last_pos;
    let mut lines: Vec<String> = Vec::new();
    while rdr.curr_is('/') && rdr.nextch_is('/') {
        let line = rdr.read_one_line_comment();
        debug!("{}", line);
        // Doc comments are not put in comments.
        if is_doc_comment(&line[..]) {
            break;
        }
        lines.push(line);
        rdr.consume_non_eol_whitespace();
    }
    debug!("<<< line comments");
    if !lines.is_empty() {
        comments.push(Comment {
            style: if code_to_the_left { Trailing } else { Isolated },
            lines: lines,
            pos: p
        });
    }
}

/// Returns None if the first col chars of s contain a non-whitespace char.
/// Otherwise returns Some(k) where k is first char offset after that leading
/// whitespace.  Note k may be outside bounds of s.
fn all_whitespace(s: &str, col: CharPos) -> Option<usize> {
    let len = s.len();
    let mut col = col.to_usize();
    let mut cursor: usize = 0;
    while col > 0 && cursor < len {
        let ch = char_at(s, cursor);
        if !ch.is_whitespace() {
            return None;
        }
        cursor += ch.len_utf8();
        col -= 1;
    }
    return Some(cursor);
}

fn trim_whitespace_prefix_and_push_line(lines: &mut Vec<String> ,
                                        s: String, col: CharPos) {
    let len = s.len();
    let s1 = match all_whitespace(&s[..], col) {
        Some(col) => {
            if col < len {
                (&s[col..len]).to_string()
            } else {
                "".to_string()
            }
        }
        None => s,
    };
    debug!("pushing line: {}", s1);
    lines.push(s1);
}

fn read_block_comment(rdr: &mut StringReader,
                      code_to_the_left: bool,
                      comments: &mut Vec<Comment> ) {
    debug!(">>> block comment");
    let p = rdr.last_pos;
    let mut lines: Vec<String> = Vec::new();
    let col = rdr.col;
    rdr.bump();
    rdr.bump();

    let mut curr_line = String::from("/*");

    // doc-comments are not really comments, they are attributes
    if (rdr.curr_is('*') && !rdr.nextch_is('*')) || rdr.curr_is('!') {
        while !(rdr.curr_is('*') && rdr.nextch_is('/')) && !rdr.is_eof() {
            curr_line.push(rdr.curr.unwrap());
            rdr.bump();
        }
        if !rdr.is_eof() {
            curr_line.push_str("*/");
            rdr.bump();
            rdr.bump();
        }
        if is_block_doc_comment(&curr_line[..]) {
            return
        }
        assert!(!curr_line.contains('\n'));
        lines.push(curr_line);
    } else {
        let mut level: isize = 1;
        while level > 0 {
            debug!("=== block comment level {}", level);
            if rdr.is_eof() {
                panic!(rdr.fatal("unterminated block comment"));
            }
            if rdr.curr_is('\n') {
                trim_whitespace_prefix_and_push_line(&mut lines,
                                                     curr_line,
                                                     col);
                curr_line = String::new();
                rdr.bump();
            } else {
                curr_line.push(rdr.curr.unwrap());
                if rdr.curr_is('/') && rdr.nextch_is('*') {
                    rdr.bump();
                    rdr.bump();
                    curr_line.push('*');
                    level += 1;
                } else {
                    if rdr.curr_is('*') && rdr.nextch_is('/') {
                        rdr.bump();
                        rdr.bump();
                        curr_line.push('/');
                        level -= 1;
                    } else { rdr.bump(); }
                }
            }
        }
        if !curr_line.is_empty() {
            trim_whitespace_prefix_and_push_line(&mut lines,
                                                 curr_line,
                                                 col);
        }
    }

    let mut style = if code_to_the_left { Trailing } else { Isolated };
    rdr.consume_non_eol_whitespace();
    if !rdr.is_eof() && !rdr.curr_is('\n') && lines.len() == 1 {
        style = Mixed;
    }
    debug!("<<< block comment");
    comments.push(Comment {style: style, lines: lines, pos: p});
}


fn consume_comment(rdr: &mut StringReader,
                   code_to_the_left: bool,
                   comments: &mut Vec<Comment> ) {
    debug!(">>> consume comment");
    if rdr.curr_is('/') && rdr.nextch_is('/') {
        read_line_comments(rdr, code_to_the_left, comments);
    } else if rdr.curr_is('/') && rdr.nextch_is('*') {
        read_block_comment(rdr, code_to_the_left, comments);
    } else if rdr.curr_is('#') && rdr.nextch_is('!') {
        read_shebang_comment(rdr, code_to_the_left, comments);
    } else { panic!(); }
    debug!("<<< consume comment");
}

#[derive(Clone)]
pub struct Literal {
    pub lit: String,
    pub pos: BytePos,
}

// it appears this function is called only from pprust... that's
// probably not a good thing.
pub fn gather_comments_and_literals(span_diagnostic: &diagnostic::SpanHandler,
                                    path: String,
                                    srdr: &mut Read)
                                 -> (Vec<Comment>, Vec<Literal>) {
    let mut src = Vec::new();
    srdr.read_to_end(&mut src).unwrap();
    let src = String::from_utf8(src).unwrap();
    let cm = CodeMap::new();
    let filemap = cm.new_filemap(path, src);
    let mut rdr = lexer::StringReader::new_raw(span_diagnostic, filemap);

    let mut comments: Vec<Comment> = Vec::new();
    let mut literals: Vec<Literal> = Vec::new();
    let mut first_read: bool = true;
    while !rdr.is_eof() {
        loop {
            let mut code_to_the_left = !first_read;
            rdr.consume_non_eol_whitespace();
            if rdr.curr_is('\n') {
                code_to_the_left = false;
                consume_whitespace_counting_blank_lines(&mut rdr, &mut comments);
            }
            while rdr.peeking_at_comment() {
                consume_comment(&mut rdr, code_to_the_left, &mut comments);
                consume_whitespace_counting_blank_lines(&mut rdr, &mut comments);
            }
            break;
        }


        let bstart = rdr.last_pos;
        rdr.next_token();
        //discard, and look ahead; we're working with internal state
        let TokenAndSpan { tok, sp } = rdr.peek();
        if tok.is_lit() {
            rdr.with_str_from(bstart, |s| {
                debug!("tok lit: {}", s);
                literals.push(Literal {lit: s.to_string(), pos: sp.lo});
            })
        } else {
            debug!("tok: {}", pprust::token_to_string(&tok));
        }
        first_read = false;
    }

    (comments, literals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_block_doc_comment_1() {
        let comment = "/**\n * Test \n **  Test\n *   Test\n*/";
        let stripped = strip_doc_comment_decoration(comment);
        assert_eq!(stripped, " Test \n*  Test\n   Test");
    }

    #[test] fn test_block_doc_comment_2() {
        let comment = "/**\n * Test\n *  Test\n*/";
        let stripped = strip_doc_comment_decoration(comment);
        assert_eq!(stripped, " Test\n  Test");
    }

    #[test] fn test_block_doc_comment_3() {
        let comment = "/**\n let a: *i32;\n *a = 5;\n*/";
        let stripped = strip_doc_comment_decoration(comment);
        assert_eq!(stripped, " let a: *i32;\n *a = 5;");
    }

    #[test] fn test_block_doc_comment_4() {
        let comment = "/*******************\n test\n *********************/";
        let stripped = strip_doc_comment_decoration(comment);
        assert_eq!(stripped, " test");
    }

    #[test] fn test_line_doc_comment() {
        let stripped = strip_doc_comment_decoration("/// test");
        assert_eq!(stripped, " test");
        let stripped = strip_doc_comment_decoration("///! test");
        assert_eq!(stripped, " test");
        let stripped = strip_doc_comment_decoration("// test");
        assert_eq!(stripped, " test");
        let stripped = strip_doc_comment_decoration("// test");
        assert_eq!(stripped, " test");
        let stripped = strip_doc_comment_decoration("///test");
        assert_eq!(stripped, "test");
        let stripped = strip_doc_comment_decoration("///!test");
        assert_eq!(stripped, "test");
        let stripped = strip_doc_comment_decoration("//test");
        assert_eq!(stripped, "test");
    }
}
