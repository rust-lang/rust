// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use ast::{MetaItem, Item, Expr,};
use codemap::Span;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::deriving::generic::*;
use ext::deriving::generic::ty::*;
use parse::token;
use ptr::P;

pub fn expand_deriving_show<F>(cx: &mut ExtCtxt,
                               span: Span,
                               mitem: &MetaItem,
                               item: &Item,
                               push: F) where
    F: FnOnce(P<Item>),
{
    // &mut ::std::fmt::Formatter
    let fmtr = Ptr(box Literal(path_std!(cx, core::fmt::Formatter)),
                   Borrowed(None, ast::MutMutable));

    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: path_std!(cx, core::fmt::Debug),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        methods: vec![
            MethodDef {
                name: "fmt",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: vec!(fmtr),
                ret_ty: Literal(path_std!(cx, core::fmt::Result)),
                attributes: Vec::new(),
                combine_substructure: combine_substructure(Box::new(|a, b, c| {
                    show_substructure(a, b, c)
                }))
            }
        ],
        associated_types: Vec::new(),
    };
    trait_def.expand(cx, mitem, item, push)
}

/// We use the debug builders to do the heavy lifting here
fn show_substructure(cx: &mut ExtCtxt, span: Span,
                     substr: &Substructure) -> P<Expr> {
    // build fmt.debug_struct(<name>).field(<fieldname>, &<fieldval>)....build()
    // or fmt.debug_tuple(<name>).field(&<fieldval>)....build()
    // based on the "shape".
    let name = match *substr.fields {
        Struct(_) => substr.type_ident,
        EnumMatching(_, v, _) => v.node.name,
        EnumNonMatchingCollapsed(..) | StaticStruct(..) | StaticEnum(..) => {
            cx.span_bug(span, "nonsensical .fields in `#[derive(Debug)]`")
        }
    };

    // We want to make sure we have the expn_id set so that we can use unstable methods
    let span = Span { expn_id: cx.backtrace(), .. span };
    let name = cx.expr_lit(span, ast::Lit_::LitStr(token::get_ident(name),
                                                   ast::StrStyle::CookedStr));
    let mut expr = substr.nonself_args[0].clone();

    match *substr.fields {
        Struct(ref fields) | EnumMatching(_, _, ref fields) => {
            if fields.is_empty() || fields[0].name.is_none() {
                // tuple struct/"normal" variant
                expr = cx.expr_method_call(span,
                                           expr,
                                           token::str_to_ident("debug_tuple"),
                                           vec![name]);

                for field in fields {
                    expr = cx.expr_method_call(span,
                                               expr,
                                               token::str_to_ident("field"),
                                               vec![cx.expr_addr_of(field.span,
                                                                    field.self_.clone())]);
                }
            } else {
                // normal struct/struct variant
                expr = cx.expr_method_call(span,
                                           expr,
                                           token::str_to_ident("debug_struct"),
                                           vec![name]);

                for field in fields {
                    let name = cx.expr_lit(field.span, ast::Lit_::LitStr(
                            token::get_ident(field.name.clone().unwrap()),
                            ast::StrStyle::CookedStr));
                    expr = cx.expr_method_call(span,
                                               expr,
                                               token::str_to_ident("field"),
                                               vec![name,
                                                    cx.expr_addr_of(field.span,
                                                                    field.self_.clone())]);
                }
            }
        }
        _ => unreachable!()
    }

    cx.expr_method_call(span,
                        expr,
                        token::str_to_ident("finish"),
                        vec![])
}
