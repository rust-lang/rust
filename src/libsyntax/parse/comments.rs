// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use codemap::{BytePos, CharPos, CodeMap, Pos};
use diagnostic;
use parse::lexer::{is_whitespace, with_str_from, Reader};
use parse::lexer::{StringReader, bump, is_eof, nextch, TokenAndSpan};
use parse::lexer::{is_line_non_doc_comment, is_block_non_doc_comment};
use parse::lexer;
use parse::token;
use parse::token::{get_ident_interner};

use std::io;
use std::str;
use std::uint;

#[deriving(Clone, Eq)]
pub enum CommentStyle {
    Isolated, // No code on either side of each line of the comment
    Trailing, // Code exists to the left of the comment
    Mixed, // Code before /* foo */ and after the comment
    BlankLine, // Just a manual blank line "\n\n", for layout
}

#[deriving(Clone)]
pub struct Comment {
    style: CommentStyle,
    lines: ~[~str],
    pos: BytePos
}

pub fn is_doc_comment(s: &str) -> bool {
    (s.starts_with("///") && !is_line_non_doc_comment(s)) ||
    s.starts_with("//!") ||
    (s.starts_with("/**") && !is_block_non_doc_comment(s)) ||
    s.starts_with("/*!")
}

pub fn doc_comment_style(comment: &str) -> ast::AttrStyle {
    assert!(is_doc_comment(comment));
    if comment.starts_with("//!") || comment.starts_with("/*!") {
        ast::AttrInner
    } else {
        ast::AttrOuter
    }
}

pub fn strip_doc_comment_decoration(comment: &str) -> ~str {
    /// remove whitespace-only lines from the start/end of lines
    fn vertical_trim(lines: ~[~str]) -> ~[~str] {
        let mut i = 0u;
        let mut j = lines.len();
        // first line of all-stars should be omitted
        if lines.len() > 0 && lines[0].chars().all(|c| c == '*') {
            i += 1;
        }
        while i < j && lines[i].trim().is_empty() {
            i += 1;
        }
        // like the first, a last line of all stars should be omitted
        if j > i && lines[j - 1].chars().skip(1).all(|c| c == '*') {
            j -= 1;
        }
        while j > i && lines[j - 1].trim().is_empty() {
            j -= 1;
        }
        return lines.slice(i, j).to_owned();
    }

    /// remove a "[ \t]*\*" block from each line, if possible
    fn horizontal_trim(lines: ~[~str]) -> ~[~str] {
        let mut i = uint::MAX;
        let mut can_trim = true;
        let mut first = true;
        for line in lines.iter() {
            for (j, c) in line.chars().enumerate() {
                if j > i || !"* \t".contains_char(c) {
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
            lines.map(|line| line.slice(i + 1, line.len()).to_owned())
        } else {
            lines
        }
    }

    // one-line comments lose their prefix
    static ONLINERS: &'static [&'static str] = &["///!", "///", "//!", "//"];
    for prefix in ONLINERS.iter() {
        if comment.starts_with(*prefix) {
            return comment.slice_from(prefix.len()).to_owned();
        }
    }

    if comment.starts_with("/*") {
        let lines = comment.slice(3u, comment.len() - 2u)
            .lines_any()
            .map(|s| s.to_owned())
            .collect::<~[~str]>();

        let lines = vertical_trim(lines);
        let lines = horizontal_trim(lines);

        return lines.connect("\n");
    }

    fail!("not a doc-comment: {}", comment);
}

fn read_to_eol(rdr: &StringReader) -> ~str {
    let mut val = ~"";
    while rdr.curr.get() != '\n' && !is_eof(rdr) {
        val.push_char(rdr.curr.get());
        bump(rdr);
    }
    if rdr.curr.get() == '\n' { bump(rdr); }
    return val;
}

fn read_one_line_comment(rdr: &StringReader) -> ~str {
    let val = read_to_eol(rdr);
    assert!((val[0] == '/' as u8 && val[1] == '/' as u8) ||
                 (val[0] == '#' as u8 && val[1] == '!' as u8));
    return val;
}

fn consume_non_eol_whitespace(rdr: &StringReader) {
    while is_whitespace(rdr.curr.get()) && rdr.curr.get() != '\n' &&
            !is_eof(rdr) {
        bump(rdr);
    }
}

fn push_blank_line_comment(rdr: &StringReader, comments: &mut ~[Comment]) {
    debug!(">>> blank-line comment");
    let v: ~[~str] = ~[];
    comments.push(Comment {
        style: BlankLine,
        lines: v,
        pos: rdr.last_pos.get(),
    });
}

fn consume_whitespace_counting_blank_lines(rdr: &StringReader,
                                           comments: &mut ~[Comment]) {
    while is_whitespace(rdr.curr.get()) && !is_eof(rdr) {
        if rdr.col.get() == CharPos(0u) && rdr.curr.get() == '\n' {
            push_blank_line_comment(rdr, &mut *comments);
        }
        bump(rdr);
    }
}


fn read_shebang_comment(rdr: &StringReader, code_to_the_left: bool,
                                            comments: &mut ~[Comment]) {
    debug!(">>> shebang comment");
    let p = rdr.last_pos.get();
    debug!("<<< shebang comment");
    comments.push(Comment {
        style: if code_to_the_left { Trailing } else { Isolated },
        lines: ~[read_one_line_comment(rdr)],
        pos: p
    });
}

fn read_line_comments(rdr: &StringReader, code_to_the_left: bool,
                                          comments: &mut ~[Comment]) {
    debug!(">>> line comments");
    let p = rdr.last_pos.get();
    let mut lines: ~[~str] = ~[];
    while rdr.curr.get() == '/' && nextch(rdr) == '/' {
        let line = read_one_line_comment(rdr);
        debug!("{}", line);
        if is_doc_comment(line) { // doc-comments are not put in comments
            break;
        }
        lines.push(line);
        consume_non_eol_whitespace(rdr);
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

// Returns None if the first col chars of s contain a non-whitespace char.
// Otherwise returns Some(k) where k is first char offset after that leading
// whitespace.  Note k may be outside bounds of s.
fn all_whitespace(s: &str, col: CharPos) -> Option<uint> {
    let len = s.len();
    let mut col = col.to_uint();
    let mut cursor: uint = 0;
    while col > 0 && cursor < len {
        let r: str::CharRange = s.char_range_at(cursor);
        if !r.ch.is_whitespace() {
            return None;
        }
        cursor = r.next;
        col -= 1;
    }
    return Some(cursor);
}

fn trim_whitespace_prefix_and_push_line(lines: &mut ~[~str],
                                        s: ~str, col: CharPos) {
    let len = s.len();
    let s1 = match all_whitespace(s, col) {
        Some(col) => {
            if col < len {
                s.slice(col, len).to_owned()
            } else {  ~"" }
        }
        None => s,
    };
    debug!("pushing line: {}", s1);
    lines.push(s1);
}

fn read_block_comment(rdr: &StringReader,
                      code_to_the_left: bool,
                      comments: &mut ~[Comment]) {
    debug!(">>> block comment");
    let p = rdr.last_pos.get();
    let mut lines: ~[~str] = ~[];
    let col: CharPos = rdr.col.get();
    bump(rdr);
    bump(rdr);

    let mut curr_line = ~"/*";

    // doc-comments are not really comments, they are attributes
    if rdr.curr.get() == '*' || rdr.curr.get() == '!' {
        while !(rdr.curr.get() == '*' && nextch(rdr) == '/') && !is_eof(rdr) {
            curr_line.push_char(rdr.curr.get());
            bump(rdr);
        }
        if !is_eof(rdr) {
            curr_line.push_str("*/");
            bump(rdr);
            bump(rdr);
        }
        if !is_block_non_doc_comment(curr_line) { return; }
        assert!(!curr_line.contains_char('\n'));
        lines.push(curr_line);
    } else {
        let mut level: int = 1;
        while level > 0 {
            debug!("=== block comment level {}", level);
            if is_eof(rdr) {
                rdr.fatal(~"unterminated block comment");
            }
            if rdr.curr.get() == '\n' {
                trim_whitespace_prefix_and_push_line(&mut lines, curr_line,
                                                     col);
                curr_line = ~"";
                bump(rdr);
            } else {
                curr_line.push_char(rdr.curr.get());
                if rdr.curr.get() == '/' && nextch(rdr) == '*' {
                    bump(rdr);
                    bump(rdr);
                    curr_line.push_char('*');
                    level += 1;
                } else {
                    if rdr.curr.get() == '*' && nextch(rdr) == '/' {
                        bump(rdr);
                        bump(rdr);
                        curr_line.push_char('/');
                        level -= 1;
                    } else { bump(rdr); }
                }
            }
        }
        if curr_line.len() != 0 {
            trim_whitespace_prefix_and_push_line(&mut lines, curr_line, col);
        }
    }

    let mut style = if code_to_the_left { Trailing } else { Isolated };
    consume_non_eol_whitespace(rdr);
    if !is_eof(rdr) && rdr.curr.get() != '\n' && lines.len() == 1u {
        style = Mixed;
    }
    debug!("<<< block comment");
    comments.push(Comment {style: style, lines: lines, pos: p});
}

fn peeking_at_comment(rdr: &StringReader) -> bool {
    return ((rdr.curr.get() == '/' && nextch(rdr) == '/') ||
         (rdr.curr.get() == '/' && nextch(rdr) == '*')) ||
         (rdr.curr.get() == '#' && nextch(rdr) == '!');
}

fn consume_comment(rdr: &StringReader,
                   code_to_the_left: bool,
                   comments: &mut ~[Comment]) {
    debug!(">>> consume comment");
    if rdr.curr.get() == '/' && nextch(rdr) == '/' {
        read_line_comments(rdr, code_to_the_left, comments);
    } else if rdr.curr.get() == '/' && nextch(rdr) == '*' {
        read_block_comment(rdr, code_to_the_left, comments);
    } else if rdr.curr.get() == '#' && nextch(rdr) == '!' {
        read_shebang_comment(rdr, code_to_the_left, comments);
    } else { fail!(); }
    debug!("<<< consume comment");
}

#[deriving(Clone)]
pub struct Literal {
    lit: ~str,
    pos: BytePos
}

// it appears this function is called only from pprust... that's
// probably not a good thing.
pub fn gather_comments_and_literals(span_diagnostic:
                                        @diagnostic::SpanHandler,
                                    path: ~str,
                                    srdr: &mut io::Reader)
                                 -> (~[Comment], ~[Literal]) {
    let src = srdr.read_to_end().unwrap();
    let src = str::from_utf8_owned(src).unwrap();
    let cm = CodeMap::new();
    let filemap = cm.new_filemap(path, src);
    let rdr = lexer::new_low_level_string_reader(span_diagnostic, filemap);

    let mut comments: ~[Comment] = ~[];
    let mut literals: ~[Literal] = ~[];
    let mut first_read: bool = true;
    while !is_eof(&rdr) {
        loop {
            let mut code_to_the_left = !first_read;
            consume_non_eol_whitespace(&rdr);
            if rdr.curr.get() == '\n' {
                code_to_the_left = false;
                consume_whitespace_counting_blank_lines(&rdr, &mut comments);
            }
            while peeking_at_comment(&rdr) {
                consume_comment(&rdr, code_to_the_left, &mut comments);
                consume_whitespace_counting_blank_lines(&rdr, &mut comments);
            }
            break;
        }


        let bstart = rdr.last_pos.get();
        rdr.next_token();
        //discard, and look ahead; we're working with internal state
        let TokenAndSpan {tok: tok, sp: sp} = rdr.peek();
        if token::is_lit(&tok) {
            with_str_from(&rdr, bstart, |s| {
                debug!("tok lit: {}", s);
                literals.push(Literal {lit: s.to_owned(), pos: sp.lo});
            })
        } else {
            debug!("tok: {}", token::to_str(get_ident_interner(), &tok));
        }
        first_read = false;
    }

    (comments, literals)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test] fn test_block_doc_comment_1() {
        let comment = "/**\n * Test \n **  Test\n *   Test\n*/";
        let stripped = strip_doc_comment_decoration(comment);
        assert_eq!(stripped, ~" Test \n*  Test\n   Test");
    }

    #[test] fn test_block_doc_comment_2() {
        let comment = "/**\n * Test\n *  Test\n*/";
        let stripped = strip_doc_comment_decoration(comment);
        assert_eq!(stripped, ~" Test\n  Test");
    }

    #[test] fn test_block_doc_comment_3() {
        let comment = "/**\n let a: *int;\n *a = 5;\n*/";
        let stripped = strip_doc_comment_decoration(comment);
        assert_eq!(stripped, ~" let a: *int;\n *a = 5;");
    }

    #[test] fn test_block_doc_comment_4() {
        let comment = "/*******************\n test\n *********************/";
        let stripped = strip_doc_comment_decoration(comment);
        assert_eq!(stripped, ~" test");
    }

    #[test] fn test_line_doc_comment() {
        let stripped = strip_doc_comment_decoration("/// test");
        assert_eq!(stripped, ~" test");
        let stripped = strip_doc_comment_decoration("///! test");
        assert_eq!(stripped, ~" test");
        let stripped = strip_doc_comment_decoration("// test");
        assert_eq!(stripped, ~" test");
        let stripped = strip_doc_comment_decoration("// test");
        assert_eq!(stripped, ~" test");
        let stripped = strip_doc_comment_decoration("///test");
        assert_eq!(stripped, ~"test");
        let stripped = strip_doc_comment_decoration("///!test");
        assert_eq!(stripped, ~"test");
        let stripped = strip_doc_comment_decoration("//test");
        assert_eq!(stripped, ~"test");
    }
}
