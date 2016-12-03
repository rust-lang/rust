// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(plugin, rustc_private)]

extern crate syntax;
extern crate syntax_pos;
extern crate rustc;

#[macro_use]
extern crate log;

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, Read};
use std::path::Path;

use syntax::parse::lexer;
use rustc::dep_graph::DepGraph;
use rustc::session::{self, config};
use rustc::middle::cstore::DummyCrateStore;

use std::rc::Rc;
use syntax::ast;
use syntax::codemap;
use syntax::parse::token::{self, BinOpToken, DelimToken, Lit, Token};
use syntax::parse::lexer::TokenAndSpan;
use syntax_pos::Pos;

use syntax::symbol::{Symbol, keywords};

fn parse_token_list(file: &str) -> HashMap<String, token::Token> {
    fn id() -> token::Token {
        Token::Ident(ast::Ident::with_empty_ctxt(keywords::Invalid.name()))
    }

    let mut res = HashMap::new();

    res.insert("-1".to_string(), Token::Eof);

    for line in file.split('\n') {
        let eq = match line.trim().rfind('=') {
            Some(val) => val,
            None => continue
        };

        let val = &line[..eq];
        let num = &line[eq + 1..];

        let tok = match val {
            "SHR"               => Token::BinOp(BinOpToken::Shr),
            "DOLLAR"            => Token::Dollar,
            "LT"                => Token::Lt,
            "STAR"              => Token::BinOp(BinOpToken::Star),
            "FLOAT_SUFFIX"      => id(),
            "INT_SUFFIX"        => id(),
            "SHL"               => Token::BinOp(BinOpToken::Shl),
            "LBRACE"            => Token::OpenDelim(DelimToken::Brace),
            "RARROW"            => Token::RArrow,
            "LIT_STR"           => Token::Literal(Lit::Str_(keywords::Invalid.name()), None),
            "DOTDOT"            => Token::DotDot,
            "MOD_SEP"           => Token::ModSep,
            "DOTDOTDOT"         => Token::DotDotDot,
            "NOT"               => Token::Not,
            "AND"               => Token::BinOp(BinOpToken::And),
            "LPAREN"            => Token::OpenDelim(DelimToken::Paren),
            "ANDAND"            => Token::AndAnd,
            "AT"                => Token::At,
            "LBRACKET"          => Token::OpenDelim(DelimToken::Bracket),
            "LIT_STR_RAW"       => Token::Literal(Lit::StrRaw(keywords::Invalid.name(), 0), None),
            "RPAREN"            => Token::CloseDelim(DelimToken::Paren),
            "SLASH"             => Token::BinOp(BinOpToken::Slash),
            "COMMA"             => Token::Comma,
            "LIFETIME"          => Token::Lifetime(
                                            ast::Ident::with_empty_ctxt(keywords::Invalid.name())),
            "CARET"             => Token::BinOp(BinOpToken::Caret),
            "TILDE"             => Token::Tilde,
            "IDENT"             => id(),
            "PLUS"              => Token::BinOp(BinOpToken::Plus),
            "LIT_CHAR"          => Token::Literal(Lit::Char(keywords::Invalid.name()), None),
            "LIT_BYTE"          => Token::Literal(Lit::Byte(keywords::Invalid.name()), None),
            "EQ"                => Token::Eq,
            "RBRACKET"          => Token::CloseDelim(DelimToken::Bracket),
            "COMMENT"           => Token::Comment,
            "DOC_COMMENT"       => Token::DocComment(keywords::Invalid.name()),
            "DOT"               => Token::Dot,
            "EQEQ"              => Token::EqEq,
            "NE"                => Token::Ne,
            "GE"                => Token::Ge,
            "PERCENT"           => Token::BinOp(BinOpToken::Percent),
            "RBRACE"            => Token::CloseDelim(DelimToken::Brace),
            "BINOP"             => Token::BinOp(BinOpToken::Plus),
            "POUND"             => Token::Pound,
            "OROR"              => Token::OrOr,
            "LIT_INTEGER"       => Token::Literal(Lit::Integer(keywords::Invalid.name()), None),
            "BINOPEQ"           => Token::BinOpEq(BinOpToken::Plus),
            "LIT_FLOAT"         => Token::Literal(Lit::Float(keywords::Invalid.name()), None),
            "WHITESPACE"        => Token::Whitespace,
            "UNDERSCORE"        => Token::Underscore,
            "MINUS"             => Token::BinOp(BinOpToken::Minus),
            "SEMI"              => Token::Semi,
            "COLON"             => Token::Colon,
            "FAT_ARROW"         => Token::FatArrow,
            "OR"                => Token::BinOp(BinOpToken::Or),
            "GT"                => Token::Gt,
            "LE"                => Token::Le,
            "LIT_BINARY"        => Token::Literal(Lit::ByteStr(keywords::Invalid.name()), None),
            "LIT_BINARY_RAW"    => Token::Literal(
                                            Lit::ByteStrRaw(keywords::Invalid.name(), 0), None),
            "QUESTION"          => Token::Question,
            "SHEBANG"           => Token::Shebang(keywords::Invalid.name()),
            _                   => continue,
        };

        res.insert(num.to_string(), tok);
    }

    debug!("Token map: {:?}", res);
    res
}

fn str_to_binop(s: &str) -> token::BinOpToken {
    match s {
        "+"     => BinOpToken::Plus,
        "/"     => BinOpToken::Slash,
        "-"     => BinOpToken::Minus,
        "*"     => BinOpToken::Star,
        "%"     => BinOpToken::Percent,
        "^"     => BinOpToken::Caret,
        "&"     => BinOpToken::And,
        "|"     => BinOpToken::Or,
        "<<"    => BinOpToken::Shl,
        ">>"    => BinOpToken::Shr,
        _       => panic!("Bad binop str `{}`", s),
    }
}

/// Assuming a string/byte string literal, strip out the leading/trailing
/// hashes and surrounding quotes/raw/byte prefix.
fn fix(mut lit: &str) -> ast::Name {
    let prefix: Vec<char> = lit.chars().take(2).collect();
    if prefix[0] == 'r' {
        if prefix[1] == 'b' {
            lit = &lit[2..]
        } else {
            lit = &lit[1..];
        }
    } else if prefix[0] == 'b' {
        lit = &lit[1..];
    }

    let leading_hashes = count(lit);

    // +1/-1 to adjust for single quotes
    Symbol::intern(&lit[leading_hashes + 1..lit.len() - leading_hashes - 1])
}

/// Assuming a char/byte literal, strip the 'b' prefix and the single quotes.
fn fixchar(mut lit: &str) -> ast::Name {
    let prefix = lit.chars().next().unwrap();
    if prefix == 'b' {
        lit = &lit[1..];
    }

    Symbol::intern(&lit[1..lit.len() - 1])
}

fn count(lit: &str) -> usize {
    lit.chars().take_while(|c| *c == '#').count()
}

fn parse_antlr_token(s: &str, tokens: &HashMap<String, token::Token>, surrogate_pairs_pos: &[usize],
                     has_bom: bool)
                     -> TokenAndSpan {
    // old regex:
    // \[@(?P<seq>\d+),(?P<start>\d+):(?P<end>\d+)='(?P<content>.+?)',<(?P<toknum>-?\d+)>,\d+:\d+]
    let start = s.find("[@").unwrap();
    let comma = start + s[start..].find(",").unwrap();
    let colon = comma + s[comma..].find(":").unwrap();
    let content_start = colon + s[colon..].find("='").unwrap();
    // Use rfind instead of find, because we don't want to stop at the content
    let content_end = content_start + s[content_start..].rfind("',<").unwrap();
    let toknum_end = content_end + s[content_end..].find(">,").unwrap();

    let start = &s[comma + 1 .. colon];
    let end = &s[colon + 1 .. content_start];
    let content = &s[content_start + 2 .. content_end];
    let toknum = &s[content_end + 3 .. toknum_end];

    let not_found = format!("didn't find token {:?} in the map", toknum);
    let proto_tok = tokens.get(toknum).expect(&not_found[..]);

    let nm = Symbol::intern(content);

    debug!("What we got: content (`{}`), proto: {:?}", content, proto_tok);

    let real_tok = match *proto_tok {
        Token::BinOp(..)           => Token::BinOp(str_to_binop(content)),
        Token::BinOpEq(..)         => Token::BinOpEq(str_to_binop(&content[..content.len() - 1])),
        Token::Literal(Lit::Str_(..), n)      => Token::Literal(Lit::Str_(fix(content)), n),
        Token::Literal(Lit::StrRaw(..), n)    => Token::Literal(Lit::StrRaw(fix(content),
                                                                             count(content)), n),
        Token::Literal(Lit::Char(..), n)      => Token::Literal(Lit::Char(fixchar(content)), n),
        Token::Literal(Lit::Byte(..), n)      => Token::Literal(Lit::Byte(fixchar(content)), n),
        Token::DocComment(..)      => Token::DocComment(nm),
        Token::Literal(Lit::Integer(..), n)   => Token::Literal(Lit::Integer(nm), n),
        Token::Literal(Lit::Float(..), n)     => Token::Literal(Lit::Float(nm), n),
        Token::Literal(Lit::ByteStr(..), n)    => Token::Literal(Lit::ByteStr(nm), n),
        Token::Literal(Lit::ByteStrRaw(..), n) => Token::Literal(Lit::ByteStrRaw(fix(content),
                                                                                count(content)), n),
        Token::Ident(..)           => Token::Ident(ast::Ident::with_empty_ctxt(nm)),
        Token::Lifetime(..)        => Token::Lifetime(ast::Ident::with_empty_ctxt(nm)),
        ref t => t.clone()
    };

    let start_offset = if real_tok == Token::Eof {
        1
    } else {
        0
    };

    let offset = if has_bom { 1 } else { 0 };

    let mut lo = start.parse::<u32>().unwrap() - start_offset - offset;
    let mut hi = end.parse::<u32>().unwrap() + 1 - offset;

    // Adjust the span: For each surrogate pair already encountered, subtract one position.
    lo -= surrogate_pairs_pos.binary_search(&(lo as usize)).unwrap_or_else(|x| x) as u32;
    hi -= surrogate_pairs_pos.binary_search(&(hi as usize)).unwrap_or_else(|x| x) as u32;

    let sp = syntax_pos::Span {
        lo: syntax_pos::BytePos(lo),
        hi: syntax_pos::BytePos(hi),
        expn_id: syntax_pos::NO_EXPANSION
    };

    TokenAndSpan {
        tok: real_tok,
        sp: sp
    }
}

fn tok_cmp(a: &token::Token, b: &token::Token) -> bool {
    match a {
        &Token::Ident(id) => match b {
                &Token::Ident(id2) => id == id2,
                _ => false
        },
        _ => a == b
    }
}

fn span_cmp(antlr_sp: codemap::Span, rust_sp: codemap::Span, cm: &codemap::CodeMap) -> bool {
    antlr_sp.expn_id == rust_sp.expn_id &&
        antlr_sp.lo.to_usize() == cm.bytepos_to_file_charpos(rust_sp.lo).to_usize() &&
        antlr_sp.hi.to_usize() == cm.bytepos_to_file_charpos(rust_sp.hi).to_usize()
}

fn main() {
    fn next(r: &mut lexer::StringReader) -> TokenAndSpan {
        use syntax::parse::lexer::Reader;
        r.next_token()
    }

    let mut args = env::args().skip(1);
    let filename = args.next().unwrap();
    if filename.find("parse-fail").is_some() {
        return;
    }

    // Rust's lexer
    let mut code = String::new();
    File::open(&Path::new(&filename)).unwrap().read_to_string(&mut code).unwrap();

    let surrogate_pairs_pos: Vec<usize> = code.chars().enumerate()
                                                     .filter(|&(_, c)| c as usize > 0xFFFF)
                                                     .map(|(n, _)| n)
                                                     .enumerate()
                                                     .map(|(x, n)| x + n)
                                                     .collect();

    let has_bom = code.starts_with("\u{feff}");

    debug!("Pairs: {:?}", surrogate_pairs_pos);

    let options = config::basic_options();
    let session = session::build_session(options, &DepGraph::new(false), None,
                                         syntax::errors::registry::Registry::new(&[]),
                                         Rc::new(DummyCrateStore));
    let filemap = session.parse_sess.codemap()
                         .new_filemap("<n/a>".to_string(), None, code);
    let mut lexer = lexer::StringReader::new(session.diagnostic(), filemap);
    let cm = session.codemap();

    // ANTLR
    let mut token_file = File::open(&Path::new(&args.next().unwrap())).unwrap();
    let mut token_list = String::new();
    token_file.read_to_string(&mut token_list).unwrap();
    let token_map = parse_token_list(&token_list[..]);

    let stdin = std::io::stdin();
    let lock = stdin.lock();
    let lines = lock.lines();
    let antlr_tokens = lines.map(|l| parse_antlr_token(l.unwrap().trim(),
                                                       &token_map,
                                                       &surrogate_pairs_pos[..],
                                                       has_bom));

    for antlr_tok in antlr_tokens {
        let rustc_tok = next(&mut lexer);
        if rustc_tok.tok == Token::Eof && antlr_tok.tok == Token::Eof {
            continue
        }

        assert!(span_cmp(antlr_tok.sp, rustc_tok.sp, cm), "{:?} and {:?} have different spans",
                rustc_tok,
                antlr_tok);

        macro_rules! matches {
            ( $($x:pat),+ ) => (
                match rustc_tok.tok {
                    $($x => match antlr_tok.tok {
                        $x => {
                            if !tok_cmp(&rustc_tok.tok, &antlr_tok.tok) {
                                // FIXME #15677: needs more robust escaping in
                                // antlr
                                warn!("Different names for {:?} and {:?}", rustc_tok, antlr_tok);
                            }
                        }
                        _ => panic!("{:?} is not {:?}", antlr_tok, rustc_tok)
                    },)*
                    ref c => assert!(c == &antlr_tok.tok, "{:?} is not {:?}", antlr_tok, rustc_tok)
                }
            )
        }

        matches!(
            Token::Literal(Lit::Byte(..), _),
            Token::Literal(Lit::Char(..), _),
            Token::Literal(Lit::Integer(..), _),
            Token::Literal(Lit::Float(..), _),
            Token::Literal(Lit::Str_(..), _),
            Token::Literal(Lit::StrRaw(..), _),
            Token::Literal(Lit::ByteStr(..), _),
            Token::Literal(Lit::ByteStrRaw(..), _),
            Token::Ident(..),
            Token::Lifetime(..),
            Token::Interpolated(..),
            Token::DocComment(..),
            Token::Shebang(..)
        );
    }
}
