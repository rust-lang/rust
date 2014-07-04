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
Syntax extension to create floating point literals from hexadecimal strings

Once loaded, hexfloat!() is called with a string containing the hexadecimal
floating-point literal, and an optional type (f32 or f64).
If the type is omitted, the literal is treated the same as a normal unsuffixed
literal.

# Examples

To load the extension and use it:

```rust,ignore
#[phase(plugin)]
extern crate hexfloat;

fn main() {
    let val = hexfloat!("0x1.ffffb4", f32);
}
```

# References

* [ExploringBinary: hexadecimal floating point constants]
  (http://www.exploringbinary.com/hexadecimal-floating-point-constants/)

*/

#![crate_id = "hexfloat#0.11.0"]
#![experimental]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/0.11.0/")]
#![feature(plugin_registrar, managed_boxes)]

extern crate syntax;
extern crate rustc;

use syntax::ast;
use syntax::codemap::{Span, mk_sp};
use syntax::ext::base;
use syntax::ext::base::{ExtCtxt, MacExpr};
use syntax::ext::build::AstBuilder;
use syntax::parse::token;
use rustc::plugin::Registry;

use std::gc::Gc;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_macro("hexfloat", expand_syntax_ext);
}

//Check if the literal is valid (as LLVM expects),
//and return a descriptive error if not.
fn hex_float_lit_err(s: &str) -> Option<(uint, String)> {
    let mut chars = s.chars().peekable();
    let mut i = 0;
    if chars.peek() == Some(&'-') { chars.next(); i+= 1 }
    if chars.next() != Some('0') {
        return Some((i, "Expected '0'".to_string()));
    } i+=1;
    if chars.next() != Some('x') {
        return Some((i, "Expected 'x'".to_string()));
    } i+=1;
    let mut d_len = 0i;
    for _ in chars.take_while(|c| c.is_digit_radix(16)) { chars.next(); i+=1; d_len += 1;}
    if chars.next() != Some('.') {
        return Some((i, "Expected '.'".to_string()));
    } i+=1;
    let mut f_len = 0i;
    for _ in chars.take_while(|c| c.is_digit_radix(16)) { chars.next(); i+=1; f_len += 1;}
    if d_len == 0 && f_len == 0 {
        return Some((i, "Expected digits before or after decimal \
                         point".to_string()));
    }
    if chars.next() != Some('p') {
        return Some((i, "Expected 'p'".to_string()));
    } i+=1;
    if chars.peek() == Some(&'-') { chars.next(); i+= 1 }
    let mut e_len = 0i;
    for _ in chars.take_while(|c| c.is_digit()) { chars.next(); i+=1; e_len += 1}
    if e_len == 0 {
        return Some((i, "Expected exponent digits".to_string()));
    }
    match chars.next() {
        None => None,
        Some(_) => Some((i, "Expected end of string".to_string()))
    }
}

pub fn expand_syntax_ext(cx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
                         -> Box<base::MacResult> {
    let (expr, ty_lit) = parse_tts(cx, tts);

    let ty = match ty_lit {
        None => None,
        Some(Ident{ident, span}) => match token::get_ident(ident).get() {
            "f32" => Some(ast::TyF32),
            "f64" => Some(ast::TyF64),
            _ => {
                cx.span_err(span, "invalid floating point type in hexfloat!");
                None
            }
        }
    };

    let s = match expr.node {
        // expression is a literal
        ast::ExprLit(lit) => match lit.node {
            // string literal
            ast::LitStr(ref s, _) => {
                s.clone()
            }
            _ => {
                cx.span_err(expr.span, "unsupported literal in hexfloat!");
                return base::DummyResult::expr(sp);
            }
        },
        _ => {
            cx.span_err(expr.span, "non-literal in hexfloat!");
            return base::DummyResult::expr(sp);
        }
    };

    {
        let err = hex_float_lit_err(s.get());
        match err {
            Some((err_pos, err_str)) => {
                let pos = expr.span.lo + syntax::codemap::Pos::from_uint(err_pos + 1);
                let span = syntax::codemap::mk_sp(pos,pos);
                cx.span_err(span,
                            format!("invalid hex float literal in hexfloat!: \
                                     {}",
                                    err_str).as_slice());
                return base::DummyResult::expr(sp);
            }
            _ => ()
        }
    }

    let lit = match ty {
        None => ast::LitFloatUnsuffixed(s),
        Some (ty) => ast::LitFloat(s, ty)
    };
    MacExpr::new(cx.expr_lit(sp, lit))
}

struct Ident {
    ident: ast::Ident,
    span: Span
}

fn parse_tts(cx: &ExtCtxt,
             tts: &[ast::TokenTree]) -> (Gc<ast::Expr>, Option<Ident>) {
    let p = &mut cx.new_parser_from_tts(tts);
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

// FIXME (10872): This is required to prevent an LLVM assert on Windows
#[test]
fn dummy_test() { }
