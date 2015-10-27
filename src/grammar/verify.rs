// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(plugin, rustc_private, str_char, collections)]

extern crate syntax;
extern crate rustc;

#[macro_use]
extern crate log;

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, Read};
use std::path::Path;

use syntax::parse;
use syntax::parse::lexer;
use rustc::session::{self, config};

use syntax::ast;
use syntax::ast::Name;
use syntax::codemap;
use syntax::codemap::Pos;
use syntax::parse::token;
use syntax::parse::lexer::TokenAndSpan;

fn parse_token_list(file: &str) -> HashMap<String, token::Token> {
    fn id() -> token::Token {
        token::Ident(ast::Ident::with_empty_ctxt(Name(0)), token::Plain)
    }

    let mut res = HashMap::new();

    res.insert("-1".to_string(), token::Eof);

    for line in file.split('\n') {
        let eq = match line.trim().rfind('=') {
            Some(val) => val,
            None => continue
        };

        let val = &line[..eq];
        let num = &line[eq + 1..];

        let tok = match val {
            "SHR"               => token::BinOp(token::Shr),
            "DOLLAR"            => token::Dollar,
            "LT"                => token::Lt,
            "STAR"              => token::BinOp(token::Star),
            "FLOAT_SUFFIX"      => id(),
            "INT_SUFFIX"        => id(),
            "SHL"               => token::BinOp(token::Shl),
            "LBRACE"            => token::OpenDelim(token::Brace),
            "RARROW"            => token::RArrow,
            "LIT_STR"           => token::Literal(token::Str_(Name(0)), None),
            "DOTDOT"            => token::DotDot,
            "MOD_SEP"           => token::ModSep,
            "DOTDOTDOT"         => token::DotDotDot,
            "NOT"               => token::Not,
            "AND"               => token::BinOp(token::And),
            "LPAREN"            => token::OpenDelim(token::Paren),
            "ANDAND"            => token::AndAnd,
            "AT"                => token::At,
            "LBRACKET"          => token::OpenDelim(token::Bracket),
            "LIT_STR_RAW"       => token::Literal(token::StrRaw(Name(0), 0), None),
            "RPAREN"            => token::CloseDelim(token::Paren),
            "SLASH"             => token::BinOp(token::Slash),
            "COMMA"             => token::Comma,
            "LIFETIME"          => token::Lifetime(ast::Ident::with_empty_ctxt(Name(0))),
            "CARET"             => token::BinOp(token::Caret),
            "TILDE"             => token::Tilde,
            "IDENT"             => id(),
            "PLUS"              => token::BinOp(token::Plus),
            "LIT_CHAR"          => token::Literal(token::Char(Name(0)), None),
            "LIT_BYTE"          => token::Literal(token::Byte(Name(0)), None),
            "EQ"                => token::Eq,
            "RBRACKET"          => token::CloseDelim(token::Bracket),
            "COMMENT"           => token::Comment,
            "DOC_COMMENT"       => token::DocComment(Name(0)),
            "DOT"               => token::Dot,
            "EQEQ"              => token::EqEq,
            "NE"                => token::Ne,
            "GE"                => token::Ge,
            "PERCENT"           => token::BinOp(token::Percent),
            "RBRACE"            => token::CloseDelim(token::Brace),
            "BINOP"             => token::BinOp(token::Plus),
            "POUND"             => token::Pound,
            "OROR"              => token::OrOr,
            "LIT_INTEGER"       => token::Literal(token::Integer(Name(0)), None),
            "BINOPEQ"           => token::BinOpEq(token::Plus),
            "LIT_FLOAT"         => token::Literal(token::Float(Name(0)), None),
            "WHITESPACE"        => token::Whitespace,
            "UNDERSCORE"        => token::Underscore,
            "MINUS"             => token::BinOp(token::Minus),
            "SEMI"              => token::Semi,
            "COLON"             => token::Colon,
            "FAT_ARROW"         => token::FatArrow,
            "OR"                => token::BinOp(token::Or),
            "GT"                => token::Gt,
            "LE"                => token::Le,
            "LIT_BYTE_STR"      => token::Literal(token::ByteStr(Name(0)), None),
            "LIT_BYTE_STR_RAW"  => token::Literal(token::ByteStrRaw(Name(0), 0), None),
            "QUESTION"          => token::Question,
            "SHEBANG"           => token::Shebang(Name(0)),
            _                   => continue,
        };

        res.insert(num.to_string(), tok);
    }

    debug!("Token map: {:?}", res);
    res
}

fn str_to_binop(s: &str) -> token::BinOpToken {
    match s {
        "+"     => token::Plus,
        "/"     => token::Slash,
        "-"     => token::Minus,
        "*"     => token::Star,
        "%"     => token::Percent,
        "^"     => token::Caret,
        "&"     => token::And,
        "|"     => token::Or,
        "<<"    => token::Shl,
        ">>"    => token::Shr,
        _       => panic!("Bad binop str `{}`", s),
    }
}

/// Assuming a string/byte string literal, strip out the leading/trailing
/// hashes and surrounding quotes/raw/byte prefix.
fn fix(mut lit: &str) -> ast::Name {
    if lit.char_at(0) == 'r' {
        if lit.char_at(1) == 'b' {
            lit = &lit[2..]
        } else {
            lit = &lit[1..];
        }
    } else if lit.char_at(0) == 'b' {
        lit = &lit[1..];
    }

    let leading_hashes = count(lit);

    // +1/-1 to adjust for single quotes
    parse::token::intern(&lit[leading_hashes + 1..lit.len() - leading_hashes - 1])
}

/// Assuming a char/byte literal, strip the 'b' prefix and the single quotes.
fn fixchar(mut lit: &str) -> ast::Name {
    if lit.char_at(0) == 'b' {
        lit = &lit[1..];
    }

    parse::token::intern(&lit[1..lit.len() - 1])
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

    let nm = parse::token::intern(content);

    debug!("What we got: content (`{}`), proto: {:?}", content, proto_tok);

    let real_tok = match *proto_tok {
        token::BinOp(..)           => token::BinOp(str_to_binop(content)),
        token::BinOpEq(..)         => token::BinOpEq(str_to_binop(&content[..content.len() - 1])),
        token::Literal(token::Str_(..), n)      => token::Literal(token::Str_(fix(content)), n),
        token::Literal(token::StrRaw(..), n)    => token::Literal(token::StrRaw(fix(content),
                                                                             count(content)), n),
        token::Literal(token::Char(..), n)      => token::Literal(token::Char(fixchar(content)), n),
        token::Literal(token::Byte(..), n)      => token::Literal(token::Byte(fixchar(content)), n),
        token::DocComment(..)      => token::DocComment(nm),
        token::Literal(token::Integer(..), n)   => token::Literal(token::Integer(nm), n),
        token::Literal(token::Float(..), n)     => token::Literal(token::Float(nm), n),
        token::Literal(token::ByteStr(..), n)    => token::Literal(token::ByteStr(nm), n),
        token::Literal(token::ByteStrRaw(..), n) => token::Literal(token::ByteStrRaw(fix(content),
                                                                                count(content)), n),
        token::Ident(..)           => token::Ident(ast::Ident::with_empty_ctxt(nm),
                                                   token::ModName),
        token::Lifetime(..)        => token::Lifetime(ast::Ident::with_empty_ctxt(nm)),
        ref t => t.clone()
    };

    let start_offset = if real_tok == token::Eof {
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

    let sp = codemap::Span {
        lo: codemap::BytePos(lo),
        hi: codemap::BytePos(hi),
        expn_id: codemap::NO_EXPANSION
    };

    TokenAndSpan {
        tok: real_tok,
        sp: sp
    }
}

fn tok_cmp(a: &token::Token, b: &token::Token) -> bool {
    match a {
        &token::Ident(id, _) => match b {
                &token::Ident(id2, _) => id == id2,
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
    let session = session::build_session(options, None,
                                         syntax::diagnostics::registry::Registry::new(&[]));
    let filemap = session.parse_sess.codemap().new_filemap(String::from("<n/a>"), code);
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
        if rustc_tok.tok == token::Eof && antlr_tok.tok == token::Eof {
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
            token::Literal(token::Byte(..), _),
            token::Literal(token::Char(..), _),
            token::Literal(token::Integer(..), _),
            token::Literal(token::Float(..), _),
            token::Literal(token::Str_(..), _),
            token::Literal(token::StrRaw(..), _),
            token::Literal(token::ByteStr(..), _),
            token::Literal(token::ByteStrRaw(..), _),
            token::Ident(..),
            token::Lifetime(..),
            token::Interpolated(..),
            token::DocComment(..),
            token::Shebang(..)
        );
    }
}
