// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
Syntax extension to generate FourCCs.

Once loaded, fourcc!() is called with a single 4-character string,
and an optional ident that is either `big`, `little`, or `target`.
The ident represents endianness, and specifies in which direction
the characters should be read. If the ident is omitted, it is assumed
to be `big`, i.e. left-to-right order. It returns a u32.

# Examples

To load the extension and use it:

```rust,ignore
#[phase(syntax)]
extern crate fourcc;

fn main() {
    let val = fourcc!("\xC0\xFF\xEE!");
    assert_eq!(val, 0xC0FFEE21u32);
    let little_val = fourcc!("foo ", little);
    assert_eq!(little_val, 0x21EEFFC0u32);
}
```

# References

* [Wikipedia: FourCC](http://en.wikipedia.org/wiki/FourCC)

*/

#![crate_id = "fourcc#0.11-pre"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://static.rust-lang.org/doc/master")]

#![deny(deprecated_owned_vector)]
#![feature(macro_registrar, managed_boxes)]

extern crate syntax;

use syntax::ast;
use syntax::ast::Name;
use syntax::attr::contains;
use syntax::codemap::{Span, mk_sp};
use syntax::ext::base;
use syntax::ext::base::{SyntaxExtension, BasicMacroExpander, NormalTT, ExtCtxt, MacExpr};
use syntax::ext::build::AstBuilder;
use syntax::parse;
use syntax::parse::token;
use syntax::parse::token::InternedString;

#[macro_registrar]
pub fn macro_registrar(register: |Name, SyntaxExtension|) {
    register(token::intern("fourcc"),
        NormalTT(box BasicMacroExpander {
            expander: expand_syntax_ext,
            span: None,
        },
        None));
}

pub fn expand_syntax_ext(cx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
                         -> Box<base::MacResult> {
    let (expr, endian) = parse_tts(cx, tts);

    let little = match endian {
        None => false,
        Some(Ident{ident, span}) => match token::get_ident(ident).get() {
            "little" => true,
            "big" => false,
            "target" => target_endian_little(cx, sp),
            _ => {
                cx.span_err(span, "invalid endian directive in fourcc!");
                target_endian_little(cx, sp)
            }
        }
    };

    let s = match expr.node {
        // expression is a literal
        ast::ExprLit(ref lit) => match lit.node {
            // string literal
            ast::LitStr(ref s, _) => {
                if s.get().char_len() != 4 {
                    cx.span_err(expr.span, "string literal with len != 4 in fourcc!");
                }
                s
            }
            _ => {
                cx.span_err(expr.span, "unsupported literal in fourcc!");
                return base::DummyResult::expr(sp)
            }
        },
        _ => {
            cx.span_err(expr.span, "non-literal in fourcc!");
            return base::DummyResult::expr(sp)
        }
    };

    let mut val = 0u32;
    for codepoint in s.get().chars().take(4) {
        let byte = if codepoint as u32 > 0xFF {
            cx.span_err(expr.span, "fourcc! literal character out of range 0-255");
            0u8
        } else {
            codepoint as u8
        };

        val = if little {
            (val >> 8) | ((byte as u32) << 24)
        } else {
            (val << 8) | (byte as u32)
        };
    }
    let e = cx.expr_lit(sp, ast::LitUint(val as u64, ast::TyU32));
    MacExpr::new(e)
}

struct Ident {
    ident: ast::Ident,
    span: Span
}

fn parse_tts(cx: &ExtCtxt, tts: &[ast::TokenTree]) -> (@ast::Expr, Option<Ident>) {
    let p = &mut parse::new_parser_from_tts(cx.parse_sess(),
                                            cx.cfg(),
                                            tts.iter()
                                               .map(|x| (*x).clone())
                                               .collect());
    let ex = p.parse_expr();
    let id = if p.token == token::EOF {
        None
    } else {
        p.expect(&token::COMMA);
        let lo = p.span.lo;
        let ident = p.parse_ident();
        let hi = p.last_span.hi;
        Some(Ident{ident: ident, span: mk_sp(lo, hi)})
    };
    if p.token != token::EOF {
        p.unexpected();
    }
    (ex, id)
}

fn target_endian_little(cx: &ExtCtxt, sp: Span) -> bool {
    let meta = cx.meta_name_value(sp, InternedString::new("target_endian"),
        ast::LitStr(InternedString::new("little"), ast::CookedStr));
    contains(cx.cfg().as_slice(), meta)
}

// FIXME (10872): This is required to prevent an LLVM assert on Windows
#[test]
fn dummy_test() { }
