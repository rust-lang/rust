// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(globs, phase, macro_rules)]

extern crate syntax;
extern crate rustc;

#[phase(link)]
extern crate regex;

#[phase(link, plugin)]
extern crate log;

#[phase(plugin)] extern crate regex_macros;

use std::collections::HashMap;
use std::io::File;

use syntax::parse;
use syntax::parse::lexer;
use rustc::driver::{session, config};

use syntax::ast;
use syntax::ast::Name;
use syntax::parse::token;
use syntax::parse::lexer::TokenAndSpan;

fn parse_token_list(file: &str) -> HashMap<String, Token> {
    fn id() -> Token {
        token::Ident(ast::Ident { name: Name(0), ctxt: 0, }, token::Plain)
    }

    let mut res = HashMap::new();

    res.insert("-1".to_string(), EOF);

    for line in file.split('\n') {
        let eq = match line.trim().rfind('=') {
            Some(val) => val,
            None => continue
        };

        let val = line.slice_to(eq);
        let num = line.slice_from(eq + 1);

        let tok = match val {
            "SHR"               => token::BinOp(token::Shr),
            "DOLLAR"            => token::Dollar,
            "LT"                => token::Lt,
            "STAR"              => token::BinOp(token::Star),
            "FLOAT_SUFFIX"      => id(),
            "INT_SUFFIX"        => id(),
            "SHL"               => token::BinOp(token::Shl),
            "LBRACE"            => token::LBrace,
            "RARROW"            => token::Rarrow,
            "LIT_STR"           => token::LitStr(Name(0)),
            "DOTDOT"            => token::DotDot,
            "MOD_SEP"           => token::ModSep,
            "DOTDOTDOT"         => token::DotDotDot,
            "NOT"               => token::Not,
            "AND"               => token::BinOp(token::And),
            "LPAREN"            => token::LParen,
            "ANDAND"            => token::AndAnd,
            "AT"                => token::At,
            "LBRACKET"          => token::LBracket,
            "LIT_STR_RAW"       => token::LitStrRaw(Name(0), 0),
            "RPAREN"            => token::RParen,
            "SLASH"             => token::BinOp(token::Slash),
            "COMMA"             => token::Comma,
            "LIFETIME"          => token::Lifetime(ast::Ident { name: Name(0), ctxt: 0 }),
            "CARET"             => token::BinOp(token::Caret),
            "TILDE"             => token::Tilde,
            "IDENT"             => token::Id(),
            "PLUS"              => token::BinOp(token::Plus),
            "LIT_CHAR"          => token::LitChar(Name(0)),
            "LIT_BYTE"          => token::LitByte(Name(0)),
            "EQ"                => token::Eq,
            "RBRACKET"          => token::RBracket,
            "COMMENT"           => token::Comment,
            "DOC_COMMENT"       => token::DocComment(Name(0)),
            "DOT"               => token::Dot,
            "EQEQ"              => token::EqEq,
            "NE"                => token::Ne,
            "GE"                => token::Ge,
            "PERCENT"           => token::BinOp(token::Percent),
            "RBRACE"            => token::RBrace,
            "BINOP"             => token::BinOp(token::Plus),
            "POUND"             => token::Pound,
            "OROR"              => token::OrOr,
            "LIT_INTEGER"       => token::LitInteger(Name(0)),
            "BINOPEQ"           => token::BinOpEq(token::Plus),
            "LIT_FLOAT"         => token::LitFloat(Name(0)),
            "WHITESPACE"        => token::Whitespace,
            "UNDERSCORE"        => token::Underscore,
            "MINUS"             => token::BinOp(token::Minus),
            "SEMI"              => token::Semi,
            "COLON"             => token::Colon,
            "FAT_ARROW"         => token::FatArrow,
            "OR"                => token::BinOp(token::Or),
            "GT"                => token::Gt,
            "LE"                => token::Le,
            "LIT_BINARY"        => token::LitBinary(Name(0)),
            "LIT_BINARY_RAW"    => token::LitBinaryRaw(Name(0), 0),
            _                   => continue,
        };

        res.insert(num.to_string(), tok);
    }

    debug!("Token map: {}", res);
    res
}

fn str_to_binop(s: &str) -> BinOpToken {
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

/// Assuming a string/binary literal, strip out the leading/trailing
/// hashes and surrounding quotes/raw/binary prefix.
fn fix(mut lit: &str) -> ast::Name {
    if lit.char_at(0) == 'r' {
        if lit.char_at(1) == 'b' {
            lit = lit.slice_from(2)
        } else {
            lit = lit.slice_from(1);
        }
    } else if lit.char_at(0) == 'b' {
        lit = lit.slice_from(1);
    }

    let leading_hashes = count(lit);

    // +1/-1 to adjust for single quotes
    parse::token::intern(lit.slice(leading_hashes + 1, lit.len() - leading_hashes - 1))
}

/// Assuming a char/byte literal, strip the 'b' prefix and the single quotes.
fn fixchar(mut lit: &str) -> ast::Name {
    if lit.char_at(0) == 'b' {
        lit = lit.slice_from(1);
    }

    parse::token::intern(lit.slice(1, lit.len() - 1))
}

fn count(lit: &str) -> uint {
    lit.chars().take_while(|c| *c == '#').count()
}

fn parse_antlr_token(s: &str, tokens: &HashMap<String, Token>) -> TokenAndSpan {
    let re = regex!(
      r"\[@(?P<seq>\d+),(?P<start>\d+):(?P<end>\d+)='(?P<content>.+?)',<(?P<toknum>-?\d+)>,\d+:\d+]"
    );

    let m = re.captures(s).expect(format!("The regex didn't match {}", s).as_slice());
    let start = m.name("start");
    let end = m.name("end");
    let toknum = m.name("toknum");
    let content = m.name("content");

    let proto_tok = tokens.find_equiv(&toknum).expect(format!("didn't find token {} in the map",
                                                              toknum).as_slice());

    let nm = parse::token::intern(content);

    debug!("What we got: content (`{}`), proto: {}", content, proto_tok);

    let real_tok = match *proto_tok {
        token::BinOp(..)           => token::BinOp(str_to_binop(content)),
        token::BinOpEq(..)         => token::BinOpEq(str_to_binop(content.slice_to(
                                                                    content.len() - 1))),
        token::LitStr(..)          => token::LitStr(fix(content)),
        token::LitStrRaw(..)       => token::LitStrRaw(fix(content), count(content)),
        token::LitChar(..)         => token::LitChar(fixchar(content)),
        token::LitByte(..)         => token::LitByte(fixchar(content)),
        token::DocComment(..)      => token::DocComment(nm),
        token::LitInteger(..)      => token::LitInteger(nm),
        token::LitFloat(..)        => token::LitFloat(nm),
        token::LitBinary(..)       => token::LitBinary(nm),
        token::LitBinaryRaw(..)    => token::LitBinaryRaw(fix(content), count(content)),
        token::Ident(..)           => token::Ident(ast::Ident { name: nm, ctxt: 0 },
                                                   token::ModName),
        token::Lifetime(..)        => token::Lifetime(ast::Ident { name: nm, ctxt: 0 }),
        ref t => t.clone()
    };

    let offset = if real_tok == EOF {
        1
    } else {
        0
    };

    let sp = syntax::codemap::Span {
        lo: syntax::codemap::BytePos(from_str::<u32>(start).unwrap() - offset),
        hi: syntax::codemap::BytePos(from_str::<u32>(end).unwrap() + 1),
        expn_id: syntax::codemap::NO_EXPANSION
    };

    TokenAndSpan {
        tok: real_tok,
        sp: sp
    }
}

fn tok_cmp(a: &Token, b: &Token) -> bool {
    match a {
        &token::Ident(id, _) => match b {
                &token::Ident(id2, _) => id == id2,
                _ => false
        },
        _ => a == b
    }
}

fn main() {
    fn next(r: &mut lexer::StringReader) -> TokenAndSpan {
        use syntax::parse::lexer::Reader;
        r.next_token()
    }

    let args = std::os::args();

    let mut token_file = File::open(&Path::new(args.get(2).as_slice()));
    let token_map = parse_token_list(token_file.read_to_string().unwrap().as_slice());

    let mut stdin = std::io::stdin();
    let mut antlr_tokens = stdin.lines().map(|l| parse_antlr_token(l.unwrap().as_slice().trim(),
                                                                   &token_map));

    let code = File::open(&Path::new(args.get(1).as_slice())).unwrap().read_to_string().unwrap();
    let options = config::basic_options();
    let session = session::build_session(options, None,
                                         syntax::diagnostics::registry::Registry::new([]));
    let filemap = parse::string_to_filemap(&session.parse_sess,
                                           code,
                                           String::from_str("<n/a>"));
    let mut lexer = lexer::StringReader::new(session.diagnostic(), filemap);

    for antlr_tok in antlr_tokens {
        let rustc_tok = next(&mut lexer);
        if rustc_tok.tok == EOF && antlr_tok.tok == EOF {
            continue
        }

        assert!(rustc_tok.sp == antlr_tok.sp, "{} and {} have different spans", rustc_tok,
                antlr_tok);

        macro_rules! matches (
            ( $($x:pat),+ ) => (
                match rustc_tok.tok {
                    $($x => match antlr_tok.tok {
                        $x => {
                            if !tok_cmp(&rustc_tok.tok, &antlr_tok.tok) {
                                // FIXME #15677: needs more robust escaping in
                                // antlr
                                warn!("Different names for {} and {}", rustc_tok, antlr_tok);
                            }
                        }
                        _ => panic!("{} is not {}", antlr_tok, rustc_tok)
                    },)*
                    ref c => assert!(c == &antlr_tok.tok, "{} is not {}", rustc_tok, antlr_tok)
                }
            )
        )

        matches!(
            LitByte(..),
            LitChar(..),
            LitInteger(..),
            LitFloat(..),
            LitStr(..),
            LitStrRaw(..),
            LitBinary(..),
            LitBinaryRaw(..),
            Ident(..),
            Lifetime(..),
            Interpolated(..),
            DocComment(..),
            Shebang(..)
        );
    }
}
