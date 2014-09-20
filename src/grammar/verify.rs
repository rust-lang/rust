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
use syntax::parse::token::*;
use syntax::parse::lexer::TokenAndSpan;

fn parse_token_list(file: &str) -> HashMap<String, Token> {
    fn id() -> Token {
        IDENT(ast::Ident { name: Name(0), ctxt: 0, }, false)
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
            "SHR" => BINOP(SHR),
            "DOLLAR" => DOLLAR,
            "LT" => LT,
            "STAR" => BINOP(STAR),
            "FLOAT_SUFFIX" => id(),
            "INT_SUFFIX" => id(),
            "SHL" => BINOP(SHL),
            "LBRACE" => LBRACE,
            "RARROW" => RARROW,
            "LIT_STR" => LIT_STR(Name(0)),
            "DOTDOT" => DOTDOT,
            "MOD_SEP" => MOD_SEP,
            "DOTDOTDOT" => DOTDOTDOT,
            "NOT" => NOT,
            "AND" => BINOP(AND),
            "LPAREN" => LPAREN,
            "ANDAND" => ANDAND,
            "AT" => AT,
            "LBRACKET" => LBRACKET,
            "LIT_STR_RAW" => LIT_STR_RAW(Name(0), 0),
            "RPAREN" => RPAREN,
            "SLASH" => BINOP(SLASH),
            "COMMA" => COMMA,
            "LIFETIME" => LIFETIME(ast::Ident { name: Name(0), ctxt: 0 }),
            "CARET" => BINOP(CARET),
            "TILDE" => TILDE,
            "IDENT" => id(),
            "PLUS" => BINOP(PLUS),
            "LIT_CHAR" => LIT_CHAR(Name(0)),
            "LIT_BYTE" => LIT_BYTE(Name(0)),
            "EQ" => EQ,
            "RBRACKET" => RBRACKET,
            "COMMENT" => COMMENT,
            "DOC_COMMENT" => DOC_COMMENT(Name(0)),
            "DOT" => DOT,
            "EQEQ" => EQEQ,
            "NE" => NE,
            "GE" => GE,
            "PERCENT" => BINOP(PERCENT),
            "RBRACE" => RBRACE,
            "BINOP" => BINOP(PLUS),
            "POUND" => POUND,
            "OROR" => OROR,
            "LIT_INTEGER" => LIT_INTEGER(Name(0)),
            "BINOPEQ" => BINOPEQ(PLUS),
            "LIT_FLOAT" => LIT_FLOAT(Name(0)),
            "WHITESPACE" => WS,
            "UNDERSCORE" => UNDERSCORE,
            "MINUS" => BINOP(MINUS),
            "SEMI" => SEMI,
            "COLON" => COLON,
            "FAT_ARROW" => FAT_ARROW,
            "OR" => BINOP(OR),
            "GT" => GT,
            "LE" => LE,
            "LIT_BINARY" => LIT_BINARY(Name(0)),
            "LIT_BINARY_RAW" => LIT_BINARY_RAW(Name(0), 0),
            _ => continue
        };

        res.insert(num.to_string(), tok);
    }

    debug!("Token map: {}", res);
    res
}

fn str_to_binop(s: &str) -> BinOp {
    match s {
        "+" => PLUS,
        "/" => SLASH,
        "-" => MINUS,
        "*" => STAR,
        "%" => PERCENT,
        "^" => CARET,
        "&" => AND,
        "|" => OR,
        "<<" => SHL,
        ">>" => SHR,
        _ => fail!("Bad binop str `{}`", s)
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
        BINOP(..) => BINOP(str_to_binop(content)),
        BINOPEQ(..) => BINOPEQ(str_to_binop(content.slice_to(content.len() - 1))),
        LIT_STR(..) => LIT_STR(fix(content)),
        LIT_STR_RAW(..) => LIT_STR_RAW(fix(content), count(content)),
        LIT_CHAR(..) => LIT_CHAR(fixchar(content)),
        LIT_BYTE(..) => LIT_BYTE(fixchar(content)),
        DOC_COMMENT(..) => DOC_COMMENT(nm),
        LIT_INTEGER(..) => LIT_INTEGER(nm),
        LIT_FLOAT(..) => LIT_FLOAT(nm),
        LIT_BINARY(..) => LIT_BINARY(nm),
        LIT_BINARY_RAW(..) => LIT_BINARY_RAW(fix(content), count(content)),
        IDENT(..) => IDENT(ast::Ident { name: nm, ctxt: 0 }, true),
        LIFETIME(..) => LIFETIME(ast::Ident { name: nm, ctxt: 0 }),
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
        &IDENT(id, _) => match b {
                &IDENT(id2, _) => id == id2,
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
                        _ => fail!("{} is not {}", antlr_tok, rustc_tok)
                    },)*
                    ref c => assert!(c == &antlr_tok.tok, "{} is not {}", rustc_tok, antlr_tok)
                }
            )
        )

        matches!(LIT_BYTE(..),
            LIT_CHAR(..),
            LIT_INTEGER(..),
            LIT_FLOAT(..),
            LIT_STR(..),
            LIT_STR_RAW(..),
            LIT_BINARY(..),
            LIT_BINARY_RAW(..),
            IDENT(..),
            LIFETIME(..),
            INTERPOLATED(..),
            DOC_COMMENT(..),
            SHEBANG(..)
        );
    }
}
