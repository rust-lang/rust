// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast::*;
use syntax::codemap::Spanned;
use syntax::ext::base::*;
use syntax::ext::build::AstBuilder;
use syntax::parse::token;
use syntax::print::pprust;
use syntax::tokenstream::{TokenStream, TokenTree};
use syntax_pos::{Span, DUMMY_SP};

pub fn expand_assert<'cx>(
    cx: &'cx mut ExtCtxt,
    sp: Span,
    tts: &[TokenTree],
) -> Box<MacResult + 'cx> {
    let mut parser = cx.new_parser_from_tts(tts);
    let cond_expr = panictry!(parser.parse_expr());
    let custom_msg_args = if parser.eat(&token::Comma) {
        let ts = parser.parse_tokens();
        if !ts.is_empty() {
            Some(ts)
        } else {
            None
        }
    } else {
        None
    };

    let sp = sp.with_ctxt(sp.ctxt().apply_mark(cx.current_expansion.mark));
    let panic_call = Mac_ {
        path: Path::from_ident(sp, Ident::from_str("panic")),
        tts: if let Some(ts) = custom_msg_args {
            ts.into()
        } else {
            // `expr_to_string` escapes the string literals with `.escape_default()`
            // which escapes all non-ASCII characters with `\u`.
            let escaped_expr = escape_format_string(&unescape_printable_unicode(
                &pprust::expr_to_string(&cond_expr),
            ));

            TokenStream::from(TokenTree::Token(
                DUMMY_SP,
                token::Literal(
                    token::Lit::Str_(Name::intern(&format!("assertion failed: {}", escaped_expr))),
                    None,
                ),
            )).into()
        },
    };
    let if_expr = cx.expr_if(
        sp,
        cx.expr(sp, ExprKind::Unary(UnOp::Not, cond_expr)),
        cx.expr(
            sp,
            ExprKind::Mac(Spanned {
                span: sp,
                node: panic_call,
            }),
        ),
        None,
    );
    MacEager::expr(if_expr)
}

/// Escapes a string for use as a formatting string.
fn escape_format_string(s: &str) -> String {
    let mut res = String::with_capacity(s.len());
    for c in s.chars() {
        res.extend(c.escape_debug());
        match c {
            '{' | '}' => res.push(c),
            _ => {}
        }
    }
    res
}

#[test]
fn test_escape_format_string() {
    assert!(escape_format_string(r"foo{}\") == r"foo{{}}\\");
}

/// Unescapes the escaped unicodes (`\u{...}`) that are printable.
fn unescape_printable_unicode(mut s: &str) -> String {
    use std::{char, u32};

    let mut res = String::with_capacity(s.len());

    loop {
        if let Some(start) = s.find(r"\u{") {
            res.push_str(&s[0..start]);
            s = &s[start..];
            s.find('}')
                .and_then(|end| {
                    let v = u32::from_str_radix(&s[3..end], 16).ok()?;
                    let c = char::from_u32(v)?;
                    // Escape unprintable characters.
                    res.extend(c.escape_debug());
                    s = &s[end + 1..];
                    Some(())
                })
                .expect("lexer should have rejected invalid escape sequences");
        } else {
            res.push_str(s);
            return res;
        }
    }
}

#[test]
fn test_unescape_printable_unicode() {
    assert!(unescape_printable_unicode(r"\u{2603}\n\u{0}") == r"☃\n\u{0}");
}
