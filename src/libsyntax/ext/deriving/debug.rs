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
use ast::{MetaItem, Expr};
use codemap::{Span, respan};
use ext::base::{ExtCtxt, Annotatable};
use ext::build::AstBuilder;
use ext::deriving::generic::*;
use ext::deriving::generic::ty::*;
use parse::token;
use ptr::P;

pub fn expand_deriving_debug(cx: &mut ExtCtxt,
                            span: Span,
                            mitem: &MetaItem,
                            item: &Annotatable,
                            push: &mut FnMut(Annotatable))
{
    // &mut ::std::fmt::Formatter
    let fmtr = Ptr(Box::new(Literal(path_std!(cx, core::fmt::Formatter))),
                   Borrowed(None, ast::MutMutable));

    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: path_std!(cx, core::fmt::Debug),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        is_unsafe: false,
        methods: vec![
            MethodDef {
                name: "fmt",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: vec!(fmtr),
                ret_ty: Literal(path_std!(cx, core::fmt::Result)),
                attributes: Vec::new(),
                is_unsafe: false,
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
    let ident = match *substr.fields {
        Struct(_) => substr.type_ident,
        EnumMatching(_, v, _) => v.node.name,
        EnumNonMatchingCollapsed(..) | StaticStruct(..) | StaticEnum(..) => {
            cx.span_bug(span, "nonsensical .fields in `#[derive(Debug)]`")
        }
    };

    // We want to make sure we have the expn_id set so that we can use unstable methods
    let span = Span { expn_id: cx.backtrace(), .. span };
    let name = cx.expr_lit(span, ast::Lit_::LitStr(ident.name.as_str(),
                                                   ast::StrStyle::CookedStr));
    let builder = token::str_to_ident("builder");
    let builder_expr = cx.expr_ident(span, builder.clone());

    let fmt = substr.nonself_args[0].clone();

    let stmts = match *substr.fields {
        Struct(ref fields) | EnumMatching(_, _, ref fields) => {
            let mut stmts = vec![];
            if fields.is_empty() || fields[0].name.is_none() {
                // tuple struct/"normal" variant
                let expr = cx.expr_method_call(span,
                                               fmt,
                                               token::str_to_ident("debug_tuple"),
                                               vec![name]);
                stmts.push(cx.stmt_let(span, true, builder, expr));

                for field in fields {
                    // Use double indirection to make sure this works for unsized types
                    let field = cx.expr_addr_of(field.span, field.self_.clone());
                    let field = cx.expr_addr_of(field.span, field);

                    let expr = cx.expr_method_call(span,
                                                   builder_expr.clone(),
                                                   token::str_to_ident("field"),
                                                   vec![field]);

                    // Use `let _ = expr;` to avoid triggering the
                    // unused_results lint.
                    stmts.push(stmt_let_undescore(cx, span, expr));
                }
            } else {
                // normal struct/struct variant
                let expr = cx.expr_method_call(span,
                                               fmt,
                                               token::str_to_ident("debug_struct"),
                                               vec![name]);
                stmts.push(cx.stmt_let(span, true, builder, expr));

                for field in fields {
                    let name = cx.expr_lit(field.span, ast::Lit_::LitStr(
                            field.name.unwrap().name.as_str(),
                            ast::StrStyle::CookedStr));

                    // Use double indirection to make sure this works for unsized types
                    let field = cx.expr_addr_of(field.span, field.self_.clone());
                    let field = cx.expr_addr_of(field.span, field);
                    let expr = cx.expr_method_call(span,
                                                   builder_expr.clone(),
                                                   token::str_to_ident("field"),
                                                   vec![name, field]);
                    stmts.push(stmt_let_undescore(cx, span, expr));
                }
            }
            stmts
        }
        _ => unreachable!()
    };

    let expr = cx.expr_method_call(span,
                                   builder_expr,
                                   token::str_to_ident("finish"),
                                   vec![]);

    let block = cx.block(span, stmts, Some(expr));
    cx.expr_block(block)
}

fn stmt_let_undescore(cx: &mut ExtCtxt,
                      sp: Span,
                      expr: P<ast::Expr>) -> P<ast::Stmt> {
    let local = P(ast::Local {
        pat: cx.pat_wild(sp),
        ty: None,
        init: Some(expr),
        id: ast::DUMMY_NODE_ID,
        span: sp,
    });
    let decl = respan(sp, ast::DeclLocal(local));
    P(respan(sp, ast::StmtDecl(P(decl), ast::DUMMY_NODE_ID)))
}
