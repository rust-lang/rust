// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[crate_id = "simd_syntax#0.10-pre"];
#[crate_type = "dylib"];
#[license = "MIT/ASL2"];
#[comment = "A parse-time library to facilitate access to SIMD types & operations"];

#[feature(macro_registrar, managed_boxes)];
#[allow(deprecated_owned_vector)];

extern crate syntax;

use syntax::ast::{Expr, ExprSwizzle, ExprSimd, DUMMY_NODE_ID, Name};
use syntax::ast::{P, Item, TokenTree, Ty, TySimd};
use syntax::{ast, ast_util};
use syntax::codemap::{respan, mk_sp};
use syntax::ext::base::{get_exprs_from_tts, SyntaxExtension, BasicMacroExpander};
use syntax::ext::base::{ExtCtxt, MacResult, MRExpr, MRItem, NormalTT};
use syntax::codemap::{Span, Spanned};
use syntax::parse::{parser, token, tts_to_parser};
use syntax::parse::token::keywords;
use syntax::parse::common::{seq_sep_trailing_allowed};
use syntax::parse::attr::ParserAttr;

use std::option::{Option, Some, None};
use std::vec_ng::Vec;

#[macro_registrar]
pub fn macro_registrar(register: |Name, SyntaxExtension|) {
    register(token::intern("gather_simd"),
             NormalTT(~BasicMacroExpander {
                expander: make_simd,
                span: None,
            },
                      None));
    register(token::intern("def_type_simd"),
             NormalTT(~BasicMacroExpander {
                expander: make_def_simd_type,
                span: None,
            },
                      None));
    register(token::intern("swizzle_simd"),
             NormalTT(~BasicMacroExpander {
                expander: make_swizzle,
                span: None,
            },
                      None));
}

fn make_simd(cx: &mut ExtCtxt, sp: Span, tts: &[TokenTree]) -> MacResult {
    let elements = match get_exprs_from_tts(cx, sp, tts) {
        Some(e) => e,
        None => {
            cx.span_err(sp, "SIMD gather with zero elements");
            return MacResult::dummy_expr(sp);
        }
    };
    MRExpr(@Expr{ id: DUMMY_NODE_ID, span: sp, node: ExprSimd(elements)})
}
fn make_def_simd_type(cx: &mut ExtCtxt, sp: Span, tts: &[TokenTree]) -> MacResult {

    let parse_simd_ty = |cx: &mut ExtCtxt,
                         parser: &mut parser::Parser,
                         sep_token: &[token::Token],
                         require_simd_ty: bool|
                     -> Option<P<Ty>> {
        if parser.eat(&token::LT) {
            let lo = parser.span.lo;
            let prim = parser.parse_ty(false);
            parser.expect(&token::COMMA);
            parser.expect(&token::DOTDOT);

            let count = {
                let mut lhs = cx.expand_expr(parser.parse_bottom_expr());
                loop {
                    if parser.token == token::GT &&
                        parser.look_ahead(1, |token| {
                            sep_token.iter().any(|t| t == token )
                        }) {
                        break;
                    } else {
                        lhs = parser.parse_more_binops(lhs, 0);
                    }
                }
                lhs
            };
            parser.expect(&token::GT);
            parser.bump();
            Some(P(Ty{ id: DUMMY_NODE_ID,
                       node: TySimd(prim, count),
                       span: mk_sp(lo, parser.span.hi),
                    }))
        } else if !require_simd_ty {
            Some(parser.parse_ty(false))
        } else {
            cx.span_err(parser.span, "expected a SIMD type");
            None
        }
    };

    let mut parser =
        tts_to_parser(cx.parse_sess(),
                      tts.to_owned().move_iter().collect(),
                      cx.cfg());

    let attrs = {
        let mut attrs = Vec::new();
        let mut cont = true;
        while cont {
            match parser.token {
                token::INTERPOLATED(token::NtAttr(attr)) => {
                    parser.bump();
                    attrs.push(*attr);
                }
                token::POUND => {
                    let lo = parser.span.lo;
                    parser.bump();
                    parser.expect(&token::LBRACKET);
                    let meta_item = parser.parse_meta_item();
                    parser.expect(&token::RBRACKET);
                    let hi = parser.span.hi;
                    attrs.push(Spanned {
                            span: mk_sp(lo, hi),
                            node: ast::Attribute_ {
                                style: ast::AttrOuter,
                                value: meta_item,
                                is_sugared_doc: false,
                            }
                        });
                }
                _ => cont = false,
            };
        }
        attrs
    };

    let vis = parser.parse_visibility();
    let ty_type;

    if parser.eat_keyword(keywords::Type) {
        ty_type = keywords::Type;
    } else if parser.eat_keyword(keywords::Struct) {
        ty_type = keywords::Struct;
    } else {
        let token_str = parser.this_token_to_str();
        parser.span_err(sp,
                        format!("expected `type` or `struct` but found `{}`",
                                token_str));
        return MacResult::dummy_expr(sp);
    }

    let ident = parser.parse_ident();
    let opt_inner;
    match ty_type {
        keywords::Type => {
            parser.expect(&token::EQ);
            opt_inner = parse_simd_ty(cx, &mut parser, [token::SEMI, token::EOF], true);
            parser.expect(&token::EOF);
        }
        keywords::Struct => {
            parser.expect(&token::LPAREN);
            opt_inner = parse_simd_ty(cx, &mut parser, [token::RPAREN], true);
            parser.eat(&token::SEMI);
            parser.expect(&token::EOF);
        }
        _ => unreachable!(),
    };
    match opt_inner {
        Some(inner) => {
            let item = Item{
                vis: vis,
                attrs: attrs,
                ident: ident,
                id: ast::DUMMY_NODE_ID,
                node: match ty_type {
                    keywords::Struct => ast::ItemStruct(@ast::StructDef {
                            fields: Vec::from_elem(1,
                                                   respan(inner.span, ast::StructField_ {
                                        kind: ast::UnnamedField,
                                        id: ast::DUMMY_NODE_ID,
                                        ty: inner,
                                        attrs: Vec::new(),
                                    })),
                            ctor_id: Some(ast::DUMMY_NODE_ID),
                        }, ast_util::empty_generics()),
                    keywords::Type => ast::ItemTy(inner, ast_util::empty_generics()),
                    _ => unreachable!(),
                },
                span: sp,
            };
            MRItem(@item)
        }
        None => MacResult::dummy_expr(sp)
    }
}
fn make_swizzle(cx: &mut ExtCtxt, sp: Span, tts: &[TokenTree]) -> MacResult {
    let mut parser = tts_to_parser(cx.parse_sess(),
                                   tts.to_owned().move_iter().collect(),
                                   cx.cfg());
    let left = parser.parse_expr();
    let opt_right = if parser.eat(&token::DOTDOT) {
        Some(parser.parse_expr())
    } else { None };
    parser.expect(&token::RARROW);
    let mask = parser.parse_unspanned_seq(&token::LPAREN,
                                          &token::RPAREN,
                                          seq_sep_trailing_allowed(token::COMMA),
                                          |s| {
            cx.expand_expr(s.parse_expr())
        });

    MRExpr(@Expr{ id: DUMMY_NODE_ID, span: sp, node: ExprSwizzle(left, opt_right, mask)})
}
